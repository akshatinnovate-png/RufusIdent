"""Multi-frame decisions at the gate.

The browser gives every person walking up a track id. Frames of the same
track are fused before anyone is marked:

  * embeddings are averaged (weighted by face quality) into one template
  * each frame also votes for its own best student
  * liveness (anti-spoof) is averaged over the frames

A student is marked only when the fused score clears the calibrated
threshold, beats the runner-up by a clear margin (twice the margin for
look-alike students), wins most of the frame votes, and the face is live.
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
    created: float
    last: float
    embs: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    lives: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    result: dict | None = None
    decided_at: float = 0.0


class Fusion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: dict[str, Track] = {}
        self.lock = threading.Lock()

    def _get(self, tid: str, now: float) -> Track:
        for k in [k for k, t in self.tracks.items() if now - t.last > self.cfg.track_ttl_s]:
            del self.tracks[k]
        t = self.tracks.get(tid)
        if t and t.result and t.result["status"] in ("unknown", "review", "spoof") and now - t.decided_at > self.cfg.retry_after_s:
            t = None   # same person still standing there: give them a fresh attempt
        if t is None:
            t = self.tracks[tid] = Track(tid, now, now)
        t.last = now
        return t

    def update(self, tid: str, emb: np.ndarray, weight: float, live: float | None, gallery: Gallery) -> dict:
        cfg = self.cfg
        now = time.monotonic()
        with self.lock:
            t = self._get(tid, now)
            if t.result:
                return {**t.result, "cached": True}

            t_accept = cfg.t_accept or gallery.t_accept
            t_review = cfg.t_review or gallery.t_review

            frame = gallery.match(emb, cfg.top_k, n=2)
            if frame and frame[0][1] >= t_review:
                t.votes[frame[0][0]] += 1
            t.embs.append(emb)
            t.weights.append(weight)
            if live is not None:
                t.lives.append(live)

            fused = np.average(np.stack(t.embs), axis=0, weights=np.asarray(t.weights))
            fused /= np.linalg.norm(fused) + 1e-9
            cands = gallery.match(fused, cfg.top_k, n=5)
            n = len(t.embs)
            live_score = float(np.mean(t.lives)) if t.lives else None

            out = {
                "frames": n,
                "need_frames": cfg.min_frames,
                "live": None if live_score is None else round(live_score, 3),
                "candidates": [{"student_id": s, "score": round(v, 4)} for s, v in cands[:3]],
                "t_accept": round(t_accept, 3),
            }
            if not cands:
                return self._decide(t, {**out, "status": "unknown", "reason": "gallery is empty"}, now)

            (best, s1) = cands[0]
            s2 = cands[1][1] if len(cands) > 1 else 0.0
            margin = s1 - s2
            lookalike = len(cands) > 1 and gallery.is_lookalike(best, cands[1][0])
            need_margin = cfg.margin * (2 if lookalike else 1)
            share = t.votes[best] / n
            out.update({"student_id": best, "score": round(s1, 4), "margin": round(margin, 4),
                        "vote_share": round(share, 3), "lookalike": lookalike,
                        "confidence": round(gallery.probability(s1), 4)})

            spoofing = live_score is not None and n >= 2 and live_score < cfg.live_threshold
            if spoofing and n >= cfg.min_frames:
                return self._decide(t, {**out, "status": "spoof", "reason": "face looks like a photo or screen"}, now)

            if n >= cfg.min_frames and not spoofing and s1 >= t_accept and margin >= need_margin and share >= cfg.vote_share:
                return self._decide(t, {**out, "status": "match"}, now)

            clearly_nobody = n >= cfg.min_frames + 1 and s1 < t_review * 0.6 and not t.votes
            if n >= cfg.max_frames or clearly_nobody:
                if spoofing:
                    return self._decide(t, {**out, "status": "spoof", "reason": "face looks like a photo or screen"}, now)
                if s1 >= t_review:
                    why = ("look-alike student" if lookalike else "too close to another student") if margin < need_margin else \
                          "frames disagree" if share < cfg.vote_share else "score below auto-accept"
                    return self._decide(t, {**out, "status": "review", "reason": why}, now)
                return self._decide(t, {**out, "status": "unknown", "reason": "not enrolled or not recognised"}, now)

            return {**out, "status": "pending"}

    def _decide(self, t: Track, result: dict, now: float) -> dict:
        t.result, t.decided_at = result, now
        return {**result, "decided": True}
