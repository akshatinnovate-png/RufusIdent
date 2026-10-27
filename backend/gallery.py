"""The enrolled faces of every student, and how a new face is matched to them.

Each enrolment photo is stored as several embeddings: the photo itself plus
simulated far-away versions of it (see engine.degrade). A photo's score is the
best of its versions, so a far, blurry gate face is compared like with like.

A student's score is the mean of their best `top_k` photo scores. One lucky
photo can't win alone, and one bad photo can't sink a student.

Thresholds are calibrated separately for near and far faces, because far
faces are naturally less certain.

The gallery also learns: very confident gate matches are kept (a few per
student) so it gets used to the real gate camera and lighting.
"""
import json
import os

import numpy as np

GALLERY_NPZ = "gallery.npz"
GALLERY_JSON = "gallery.json"
ADAPTIVE_NPZ = "adaptive.npz"

DEFAULTS = {
    "near": {"t_accept": 0.45, "t_review": 0.30, "a": 25.0, "b": -9.0},
    "far": {"t_accept": 0.50, "t_review": 0.32, "a": 25.0, "b": -10.0},
}


class Gallery:
    def __init__(self):
        self.emb = np.zeros((0, 512), np.float32)
        self.row_variant = np.zeros(0, np.int8)    # 0 = the photo itself, 1.. = far versions
        self.photo_start = np.zeros(0, np.int64)   # photo -> first row (rows grouped by photo)
        self.photo_owner = np.zeros(0, np.int32)   # photo -> student index
        self.ids: list[str] = []                   # student index -> admission no
        self.sp_start = np.zeros(0, np.int64)      # student -> first photo
        self.sp_count = np.zeros(0, np.int64)
        self.centroid = np.zeros((0, 512), np.float32)
        self.best_photo: dict[str, str] = {}
        self.lookalikes: dict[str, list[str]] = {}
        self.calib = {"source": "defaults", **{k: dict(v) for k, v in DEFAULTS.items()}}
        self.index: dict[str, int] = {}
        # learned from the gate camera
        self.adapt = np.zeros((0, 512), np.float32)
        self.adapt_owner = np.zeros(0, np.int32)
        self._adapt_rows: dict[int, np.ndarray] = {}

    # ───── build / persist ─────

    @classmethod
    def build(cls, per_student: dict[str, list[np.ndarray]], best_photo: dict[str, str]) -> "Gallery":
        """per_student[sid] = list of photos, each an array (variants x 512), variant 0 = the photo."""
        g = cls()
        g.ids = sorted(per_student, key=lambda s: (len(s), s))
        rows, variant, pstart, powner, spstart, spcount = [], [], [], [], [], []
        n_rows = 0
        for i, sid in enumerate(g.ids):
            spstart.append(len(pstart)); spcount.append(len(per_student[sid]))
            for photo in per_student[sid]:
                photo = np.atleast_2d(np.asarray(photo, np.float32))
                pstart.append(n_rows); powner.append(i)
                rows.append(photo); variant += list(range(len(photo)))
                n_rows += len(photo)
        g.emb = np.concatenate(rows) if rows else np.zeros((0, 512), np.float32)
        g.row_variant = np.asarray(variant, np.int8)
        g.photo_start, g.photo_owner = np.asarray(pstart, np.int64), np.asarray(powner, np.int32)
        g.sp_start, g.sp_count = np.asarray(spstart, np.int64), np.asarray(spcount, np.int64)
        g.best_photo = best_photo
        g._finish()
        return g

    def _finish(self):
        self.index = {sid: i for i, sid in enumerate(self.ids)}
        self.photo_end = np.append(self.photo_start[1:], len(self.emb)).astype(np.int64)
        c = np.zeros((len(self.ids), 512), np.float32)
        base = self.emb[self.photo_start] if len(self.photo_start) else np.zeros((0, 512), np.float32)
        np.add.at(c, self.photo_owner, base)
        self.centroid = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)

    def save(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(os.path.join(data_dir, GALLERY_NPZ), emb=self.emb.astype(np.float16), row_variant=self.row_variant,
                            photo_start=self.photo_start, photo_owner=self.photo_owner, sp_start=self.sp_start, sp_count=self.sp_count)
        with open(os.path.join(data_dir, GALLERY_JSON), "w", encoding="utf-8") as f:
            json.dump({"version": 2, "ids": self.ids, "best_photo": self.best_photo, "lookalikes": self.lookalikes, "calib": self.calib}, f, indent=1)

    @classmethod
    def load(cls, data_dir: str) -> "Gallery | None":
        npz, js = os.path.join(data_dir, GALLERY_NPZ), os.path.join(data_dir, GALLERY_JSON)
        if not (os.path.exists(npz) and os.path.exists(js)):
            return None
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != 2:
            return None          # made by an older version: run enroll.py again
        g = cls()
        z = np.load(npz)
        g.emb = z["emb"].astype(np.float32)
        g.emb /= np.linalg.norm(g.emb, axis=1, keepdims=True) + 1e-9
        g.row_variant, g.photo_start, g.photo_owner = z["row_variant"], z["photo_start"], z["photo_owner"]
        g.sp_start, g.sp_count = z["sp_start"], z["sp_count"]
        g.ids, g.best_photo, g.lookalikes, g.calib = meta["ids"], meta["best_photo"], meta["lookalikes"], meta["calib"]
        g._finish()
        g.load_adaptive(data_dir)
        return g

    # ───── learning from the gate ─────

    def load_adaptive(self, data_dir: str):
        path = os.path.join(data_dir, ADAPTIVE_NPZ)
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            keep = [i for i, sid in enumerate(z["owner_ids"]) if str(sid) in self.index]
            self.adapt = z["emb"][keep].astype(np.float32)
            self.adapt_owner = np.asarray([self.index[str(z["owner_ids"][i])] for i in keep], np.int32)
        self._index_adaptive()

    def save_adaptive(self, data_dir: str):
        np.savez_compressed(os.path.join(data_dir, ADAPTIVE_NPZ), emb=self.adapt.astype(np.float16),
                            owner_ids=np.asarray([self.ids[i] for i in self.adapt_owner], dtype=str))

    def add_adaptive(self, sid: str, emb: np.ndarray, per_student: int):
        s = self.index[sid]
        mine = np.where(self.adapt_owner == s)[0]
        if len(mine) >= per_student:                      # keep the newest few
            drop = mine[: len(mine) - per_student + 1]
            keep = np.setdiff1d(np.arange(len(self.adapt)), drop)
            self.adapt, self.adapt_owner = self.adapt[keep], self.adapt_owner[keep]
        self.adapt = np.vstack([self.adapt, emb[None].astype(np.float32)])
        self.adapt_owner = np.append(self.adapt_owner, s).astype(np.int32)
        self._index_adaptive()

    def _index_adaptive(self):
        self._adapt_rows = {}
        for r, s in enumerate(self.adapt_owner):
            self._adapt_rows.setdefault(int(s), []).append(r)
        self._adapt_rows = {k: np.asarray(v) for k, v in self._adapt_rows.items()}

    # ───── matching ─────

    def _photo_scores(self, sims: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(sims, self.photo_start) if len(self.photo_start) else sims[:0]

    def _student_score(self, ps: np.ndarray, s: int, k: int, asims: np.ndarray | None, exclude_photo: int = -1):
        seg = ps[self.sp_start[s]:self.sp_start[s] + self.sp_count[s]]
        if exclude_photo >= 0:
            seg = np.delete(seg, exclude_photo - self.sp_start[s])
        base = float(seg.max()) if seg.size else -1.0
        if asims is not None and s in self._adapt_rows:
            seg = np.concatenate([seg, asims[self._adapt_rows[s]]])
        if not seg.size:
            return -1.0, base
        kk = min(k, seg.size)
        return float(np.partition(seg, seg.size - kk)[-kk:].mean()), base

    def match(self, q: np.ndarray, k: int = 3, n: int = 5, shortlist: int = 400) -> list[tuple[str, float, float]]:
        """Best `n` students for a unit-length embedding: (admission_no, score, best enrolment-photo score)."""
        if not len(self.ids):
            return []
        ps = self._photo_scores(self.emb @ q)
        m = min(shortlist, ps.size)
        cand = set(np.unique(self.photo_owner[np.argpartition(-ps, m - 1)[:m]]).tolist())
        asims = None
        if len(self.adapt):
            asims = self.adapt @ q
            top = np.argsort(-asims)[:20]
            cand |= set(self.adapt_owner[top].tolist())
        scored = [(self.ids[s], *self._student_score(ps, s, k, asims)) for s in cand]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def thresholds(self, near: bool) -> dict:
        return self.calib["near" if near else "far"]

    def probability(self, score: float, near: bool = True) -> float:
        """Calibrated P(this really is the student) for a score."""
        c = self.thresholds(near)
        return float(1 / (1 + np.exp(-(c["a"] * score + c["b"]))))

    def is_lookalike(self, a: str, b: str) -> bool:
        return b in self.lookalikes.get(a, ())

    # ───── self-calibration from the enrolment photos ─────

    def calibrate(self, k: int, pair_threshold: float) -> dict:
        """Every stored embedding is used as a probe:
        genuine  = its own student's *other* photos (leave the whole photo out),
        impostor = the best-scoring other student (look-alike pairs excluded:
                   they get a stricter margin at the gate instead).
        Near thresholds come from the photos themselves, far thresholds from
        their far versions."""
        n_students = len(self.ids)
        chunk = 2048

        self.lookalikes = {}
        for lo in range(0, n_students, chunk):
            S = self.centroid[lo:lo + chunk] @ self.centroid.T
            for r, c in zip(*np.where(S > pair_threshold)):
                if lo + r != c:
                    self.lookalikes.setdefault(self.ids[lo + r], []).append(self.ids[c])

        row_photo = np.repeat(np.arange(len(self.photo_start)), self.photo_end - self.photo_start)
        stats = {"near": ([], []), "far": ([], [])}
        for lo in range(0, len(self.emb), chunk):
            E = self.emb[lo:lo + chunk]
            csims = E @ self.centroid.T
            own = self.photo_owner[row_photo[lo:lo + chunk]]
            csims[np.arange(len(E)), own] = -9
            top_other = np.argpartition(-csims, min(4, n_students - 1), axis=1)[:, :5] if n_students > 1 else None
            for r in range(len(E)):
                row, s = lo + r, own[r]
                gen, imp = stats["near" if self.row_variant[row] == 0 else "far"]
                rows = slice(self.photo_start[self.sp_start[s]], self.photo_end[self.sp_start[s] + self.sp_count[s] - 1])
                ps_own = np.maximum.reduceat(self.emb[rows] @ E[r], self.photo_start[self.sp_start[s]:self.sp_start[s] + self.sp_count[s]] - rows.start)
                ps_own = np.delete(ps_own, row_photo[row] - self.sp_start[s])
                if ps_own.size:
                    kk = min(k, ps_own.size)
                    gen.append(np.partition(ps_own, ps_own.size - kk)[-kk:].mean())
                if top_other is None:
                    continue
                best, best_c = -1.0, -1
                for c in np.unique(top_other[r]):
                    if c == s:
                        continue
                    crow = slice(self.photo_start[self.sp_start[c]], self.photo_end[self.sp_start[c] + self.sp_count[c] - 1])
                    ps = np.maximum.reduceat(self.emb[crow] @ E[r], self.photo_start[self.sp_start[c]:self.sp_start[c] + self.sp_count[c]] - crow.start)
                    kk = min(k, ps.size)
                    v = float(np.partition(ps, ps.size - kk)[-kk:].mean())
                    if v > best:
                        best, best_c = v, c
                if best_c >= 0 and not self.is_lookalike(self.ids[s], self.ids[best_c]):
                    imp.append(best)

        report = {"lookalike_students": len(self.lookalikes)}
        calibrated = False
        for level, (gen, imp) in stats.items():
            gen, imp = np.asarray(gen), np.asarray(imp)
            if gen.size >= 50 and imp.size >= 200:
                t_acc = float(max(0.38, np.quantile(imp, 0.9999) + 0.03))
                t_rev = float(max(0.25, min(t_acc - 0.1, np.quantile(imp, 0.999))))
                a, b = _logistic(np.concatenate([gen, imp]), np.concatenate([np.ones_like(gen), np.zeros_like(imp)]))
                self.calib[level] = {"t_accept": t_acc, "t_review": t_rev, "a": a, "b": b}
                calibrated = True
            else:
                self.calib[level] = dict(DEFAULTS[level])
            c = self.calib[level]
            report[level] = {
                **{kk: round(v, 4) for kk, v in c.items()},
                "genuine_n": int(gen.size), "impostor_n": int(imp.size),
                "genuine_accept_rate": float((gen >= c["t_accept"]).mean()) if gen.size else None,
                "impostor_pass_rate": float((imp >= c["t_accept"]).mean()) if imp.size else None,
                "impostor_max": float(imp.max()) if imp.size else None,
            }
        self.calib["source"] = "calibrated" if calibrated else "defaults (too few photos to calibrate)"
        report["source"] = self.calib["source"]
        if (report["near"]["genuine_accept_rate"] or 1) < 0.9:
            report["warning"] = ("Many genuine photos score below the accept threshold: photos may be low quality "
                                 "or too varied. Those students will often go to Review instead of auto-marking.")
        return report


def _logistic(x: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """1-D logistic regression (Newton's method), classes balanced."""
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    a, b = 10.0, -4.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([(w * (p - y) * x).sum(), (w * (p - y)).sum()])
        h = w * p * (1 - p)
        H = np.array([[(h * x * x).sum(), (h * x).sum()], [(h * x).sum(), h.sum()]]) + np.eye(2) * 1e-6
        step = np.linalg.solve(H, g)
        if not np.all(np.isfinite(step)):
            break
        a, b = float(np.clip(a - step[0], 1, 200)), float(np.clip(b - step[1], -200, 200))
        if np.abs(step).max() < 1e-6:
            break
    return a, b
