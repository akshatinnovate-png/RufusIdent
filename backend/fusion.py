"""Multi-frame decisions at the gate, one per tracked person.

Frames of the same person are fused before anyone is marked:

  * embeddings are averaged, weighted by face quality (size, sharpness, pose),
    so the frames taken as a student walks closer count the most
  * every frame also votes for its own best student
  * liveness (anti-spoof) is averaged over the frames big enough to judge

A student is marked only when the fused score clears the calibrated threshold
for that face size (near or far), beats the runner-up by a clear margin
(twice the margin for look-alike students), wins most of the frame votes,
matches at least one real enrolment photo, and the face was checked live.

If a new frame looks nothing like the track so far, the tracker has swapped
two people: the track's evidence is thrown away and it starts again.
"""
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from config import Config
from gallery import Gallery


@dataclass
class Track:
    tid: str
    last: float
    embs: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    widths: list = field(default_factory=list)
    lives: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    result: dict | None = None
    decided_at: float = 0.0

    def reset(self):
        self.embs, self.weights, self.widths, self.lives, self.votes = [], [], [], [], Counter()


class Fusion:
    def __init__(self, cfg: Config, liveness: bool = True):
        self.cfg = cfg
        self.liveness = liveness            # False when anti-spoof is off or its models are missing
        self.tracks: dict[str, Track] = {}
        self.lock = threading.Lock()

    def _get(self, tid: str, now: float) -> Track:
        for k in [k for k, t in self.tracks.items() if now - t.last > self.cfg.track_ttl_s]:
            del self.tracks[k]
        t = self.tracks.get(tid)
        if t and t.result and t.result["status"] in ("unknown", "review", "spoof") and now - t.decided_at > self.cfg.retry_after_s:
            t = None   # same person still there: give them a fresh attempt
        if t is None:
            t = self.tracks[tid] = Track(tid, now)
        t.last = now
        return t

    def cached(self, tid: str) -> dict | None:
        """A decision that still stands for this track (no need to look at the face again)."""
        t = self.tracks.get(tid)
        if not t or not t.result:
            return None
        if t.result["status"] == "match" or time.monotonic() - t.decided_at <= self.cfg.retry_after_s:
            t.last = time.monotonic()
            return t.result
        return None

    def update(self, tid: str, emb: np.ndarray, weight: float, width: float, live: float | None, gallery: Gallery) -> dict:
        cfg = self.cfg
        now = time.monotonic()
        with self.lock:
            t = self._get(tid, now)
            if t.result:
                return {**t.result, "cached": True}

            if t.embs:
                prev = np.average(np.stack(t.embs), axis=0, weights=np.asarray(t.weights))
                if float(prev @ emb) / (np.linalg.norm(prev) + 1e-9) < cfg.switch_guard:
                    t.reset()

            frame = gallery.match(emb, cfg.top_k, n=1)
            near_frame = width >= cfg.near_px
            if frame and frame[0][1] >= gallery.thresholds(near_frame)["t_review"]:
                t.votes[frame[0][0]] += 1
            t.embs.append(emb); t.weights.append(weight); t.widths.append(width)
            if live is not None:
                t.lives.append(live)

            fused = np.average(np.stack(t.embs), axis=0, weights=np.asarray(t.weights))
            fused /= np.linalg.norm(fused) + 1e-9
            cands = gallery.match(fused, cfg.top_k, n=5)
            n = len(t.embs)
            eff_width = float(np.average(t.widths, weights=t.weights))
            near = eff_width >= cfg.near_px
            th = gallery.thresholds(near)
            t_accept = cfg.t_accept or th["t_accept"]
            t_review = cfg.t_review or th["t_review"]
            need = cfg.min_frames if near else cfg.min_frames_far
            live_score = float(np.mean(t.lives)) if t.lives else None

            out = {
                "frames": n, "need_frames": need, "range": "near" if near else "far",
                "live": None if live_score is None else round(live_score, 3),
                "candidates": [{"student_id": s, "score": round(v, 4)} for s, v, _ in cands[:3]],
                "t_accept": round(t_accept, 3),
            }
            if not cands:
                return self._decide(t, {**out, "status": "unknown", "reason": "gallery is empty"}, now)

            best, s1, base = cands[0]
            s2 = cands[1][1] if len(cands) > 1 else 0.0
            margin = s1 - s2
            lookalike = len(cands) > 1 and gallery.is_lookalike(best, cands[1][0])
            need_margin = cfg.margin * (2 if lookalike else 1)
            share = t.votes[best] / n
            out.update({"student_id": best, "score": round(s1, 4), "margin": round(margin, 4),
                        "vote_share": round(share, 3), "lookalike": lookalike,
                        "confidence": round(gallery.probability(s1, near), 4)})

            spoofing = live_score is not None and len(t.lives) >= 2 and live_score < cfg.live_threshold
            if spoofing and len(t.lives) >= cfg.min_frames:
                return self._decide(t, {**out, "status": "spoof", "reason": "face looks like a photo or screen"}, now)
            live_ok = not spoofing and (not (cfg.require_live and self.liveness) or (live_score is not None and live_score >= cfg.live_threshold))

            # Fast track: the evidence is overwhelming, so fewer frames are needed.
            # Never for look-alikes, far faces, or without a strong liveness check.
            if cfg.fast_track and near and not lookalike and live_score is not None and live_score >= 0.85 and base >= t_accept:
                if s1 >= t_accept + 0.20 and margin >= max(0.25, 3 * need_margin):
                    need = 1
                elif s1 >= t_accept + 0.10 and margin >= 2 * need_margin:
                    need = min(need, 2)
                out["need_frames"] = need

            confident = n >= need and s1 >= t_accept and base >= t_review and margin >= need_margin and share >= cfg.vote_share
            if confident and live_ok:
                result = {**out, "status": "match"}
                # Very sure, close and live: remember this look of the student for next time.
                if cfg.adaptive and near and s1 >= t_accept + 0.08 and margin >= 2 * need_margin and max(t.widths) >= 80:
                    i = int(np.argmax(np.asarray(t.weights) * (np.asarray(t.widths) >= 80)))
                    result["_learn"] = t.embs[i]
                return self._decide(t, result, now)
            if confident and not live_ok:
                # Recognised, but too far to check it's a live face: wait for them to walk closer.
                return {**out, "status": "pending", "reason": "come closer for the liveness check"}

            # Near, clear face that isn't close to anyone: no point waiting. (Far faces
            # keep going: their scores rise as the student walks closer.)
            hopeless = near and n >= need + 1 and s1 < t_review and not t.votes
            if n >= cfg.max_frames or hopeless:
                if spoofing:
                    return self._decide(t, {**out, "status": "spoof", "reason": "face looks like a photo or screen"}, now)
                if s1 >= t_review:
                    why = out.get("reason") or (("look-alike student" if lookalike else "too close to another student") if margin < need_margin
                                                else "frames disagree" if share < cfg.vote_share else "score below auto-accept")
                    return self._decide(t, {**out, "status": "review", "reason": why}, now)
                return self._decide(t, {**out, "status": "unknown", "reason": "not enrolled or not recognised"}, now)

            return {**out, "status": "pending"}

    def _decide(self, t: Track, result: dict, now: float) -> dict:
        learn = result.pop("_learn", None)
        t.result, t.decided_at = result, now
        out = {**result, "decided": True}
        if learn is not None:
            out["_learn"] = learn
        return out
