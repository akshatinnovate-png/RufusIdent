"""The enrolled faces of every student, and how a new face is matched to them.

A student's score for a probe face is the mean of its best `top_k` matches
among that student's photos. One lucky photo can't win alone, and one bad
photo can't sink a student.
"""
import json
import os

import numpy as np

GALLERY_NPZ = "gallery.npz"
GALLERY_JSON = "gallery.json"

DEFAULT_ACCEPT = 0.45
DEFAULT_REVIEW = 0.30


class Gallery:
    def __init__(self):
        self.emb = np.zeros((0, 512), np.float32)
        self.owner = np.zeros(0, np.int32)       # row -> student index
        self.ids: list[str] = []                  # student index -> admission no
        self.start = np.zeros(0, np.int64)        # student index -> first row (rows are grouped)
        self.count = np.zeros(0, np.int64)
        self.centroid = np.zeros((0, 512), np.float32)
        self.best_photo: dict[str, str] = {}
        self.lookalikes: dict[str, list[str]] = {}
        self.calib = {"t_accept": DEFAULT_ACCEPT, "t_review": DEFAULT_REVIEW, "a": 25.0, "b": -9.0, "source": "defaults"}
        self.index = {}

    # ───── build / persist ─────

    @classmethod
    def build(cls, per_student: dict[str, list[np.ndarray]], best_photo: dict[str, str]) -> "Gallery":
        g = cls()
        g.ids = sorted(per_student, key=lambda s: (len(s), s))
        rows, owner, start, count = [], [], [], []
        for i, sid in enumerate(g.ids):
            e = np.asarray(per_student[sid], np.float32)
            start.append(sum(count)); count.append(len(e))
            rows.append(e); owner += [i] * len(e)
        g.emb = np.concatenate(rows) if rows else np.zeros((0, 512), np.float32)
        g.owner = np.asarray(owner, np.int32)
        g.start, g.count = np.asarray(start), np.asarray(count)
        g.best_photo = best_photo
        g._finish()
        return g

    def _finish(self):
        self.index = {sid: i for i, sid in enumerate(self.ids)}
        c = np.zeros((len(self.ids), 512), np.float32)
        np.add.at(c, self.owner, self.emb)
        self.centroid = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)

    def save(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(os.path.join(data_dir, GALLERY_NPZ), emb=self.emb.astype(np.float16), owner=self.owner, start=self.start, count=self.count)
        with open(os.path.join(data_dir, GALLERY_JSON), "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "best_photo": self.best_photo, "lookalikes": self.lookalikes, "calib": self.calib}, f, indent=1)

    @classmethod
    def load(cls, data_dir: str) -> "Gallery | None":
        npz, js = os.path.join(data_dir, GALLERY_NPZ), os.path.join(data_dir, GALLERY_JSON)
        if not (os.path.exists(npz) and os.path.exists(js)):
            return None
        g = cls()
        z = np.load(npz)
        g.emb = z["emb"].astype(np.float32)
        g.emb /= np.linalg.norm(g.emb, axis=1, keepdims=True) + 1e-9
        g.owner, g.start, g.count = z["owner"], z["start"], z["count"]
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
        g.ids, g.best_photo, g.lookalikes, g.calib = meta["ids"], meta["best_photo"], meta["lookalikes"], meta["calib"]
        g._finish()
        return g

    # ───── matching ─────

    def student_scores(self, sims: np.ndarray, students: np.ndarray, k: int) -> np.ndarray:
        out = np.empty(len(students), np.float32)
        for j, s in enumerate(students):
            seg = sims[self.start[s]:self.start[s] + self.count[s]]
            kk = min(k, seg.size)
            out[j] = np.partition(seg, seg.size - kk)[-kk:].mean()
        return out

    def match(self, q: np.ndarray, k: int = 3, n: int = 5, shortlist: int = 400) -> list[tuple[str, float]]:
        """Best `n` students for a unit-length embedding, as (admission_no, score)."""
        if not len(self.ids):
            return []
        sims = self.emb @ q
        m = min(shortlist, sims.size)
        rows = np.argpartition(-sims, m - 1)[:m]
        students = np.unique(self.owner[rows])
        scores = self.student_scores(sims, students, k)
        order = np.argsort(-scores)[:n]
        return [(self.ids[students[i]], float(scores[i])) for i in order]

    def probability(self, score: float) -> float:
        """Calibrated P(this really is the student) for a score."""
        return float(1 / (1 + np.exp(-(self.calib["a"] * score + self.calib["b"]))))

    def is_lookalike(self, a: str, b: str) -> bool:
        return b in self.lookalikes.get(a, ())

    @property
    def t_accept(self) -> float:
        return self.calib["t_accept"]

    @property
    def t_review(self) -> float:
        return self.calib["t_review"]

    # ───── self-calibration from the enrolment photos ─────

    def calibrate(self, k: int, pair_threshold: float) -> dict:
        """Each photo is used as a probe against everyone else:
        genuine = its own student's other photos (leave-one-out),
        impostor = the best-scoring *other* student.
        Thresholds are set so an impostor almost never passes."""
        n_students = len(self.ids)
        genuine, impostor = [], []
        chunk = 2048
        # Look-alike students (siblings, twins): compared more strictly at the gate.
        self.lookalikes = {}
        if n_students > 1:
            for lo in range(0, n_students, chunk):
                S = self.centroid[lo:lo + chunk] @ self.centroid.T
                for r, c in zip(*np.where(S > pair_threshold)):
                    a, b = lo + r, c
                    if a != b:
                        self.lookalikes.setdefault(self.ids[a], []).append(self.ids[b])

        # Impostor scores from look-alike pairs are excluded from the threshold:
        # those students are handled by a stricter margin at the gate instead.
        for lo in range(0, len(self.emb), chunk):
            E = self.emb[lo:lo + chunk]
            csims = E @ self.centroid.T                                   # shortlist by centroid
            own = self.owner[lo:lo + chunk]
            csims[np.arange(len(E)), own] = -9
            top_other = np.argpartition(-csims, min(4, n_students - 1), axis=1)[:, :5] if n_students > 1 else None
            for r in range(len(E)):
                row = lo + r
                s = own[r]
                seg = self.emb[self.start[s]:self.start[s] + self.count[s]] @ E[r]
                seg = np.delete(seg, row - self.start[s])
                if seg.size:
                    kk = min(k, seg.size)
                    genuine.append(np.partition(seg, seg.size - kk)[-kk:].mean())
                if top_other is not None:
                    best, best_c = -1.0, -1
                    for c in np.unique(top_other[r]):
                        if c == s:
                            continue
                        seg = self.emb[self.start[c]:self.start[c] + self.count[c]] @ E[r]
                        kk = min(k, seg.size)
                        v = float(np.partition(seg, seg.size - kk)[-kk:].mean())
                        if v > best:
                            best, best_c = v, c
                    if best_c >= 0 and not self.is_lookalike(self.ids[s], self.ids[best_c]):
                        impostor.append(best)
        genuine, impostor = np.asarray(genuine), np.asarray(impostor)

        report = {"genuine_n": int(genuine.size), "impostor_n": int(impostor.size)}
        if genuine.size >= 50 and impostor.size >= 200:
            t_acc = float(max(DEFAULT_REVIEW + 0.05, np.quantile(impostor, 0.9999) + 0.03, 0.38))
            t_rev = float(max(0.25, min(t_acc - 0.1, np.quantile(impostor, 0.999))))
            a, b = _logistic(np.concatenate([genuine, impostor]), np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)]))
            self.calib = {"t_accept": t_acc, "t_review": t_rev, "a": a, "b": b, "source": "calibrated"}
        else:
            self.calib = {"t_accept": DEFAULT_ACCEPT, "t_review": DEFAULT_REVIEW, "a": 25.0, "b": -9.0, "source": "defaults (too few photos to calibrate)"}
        report.update(self.calib)
        if genuine.size:
            report["genuine_accept_rate"] = float((genuine >= self.t_accept).mean())
            report["genuine_median"] = float(np.median(genuine))
        if impostor.size:
            report["impostor_pass_rate"] = float((impostor >= self.t_accept).mean())
            report["impostor_max"] = float(impostor.max())
        report["lookalike_students"] = len(self.lookalikes)
        if report.get("genuine_accept_rate", 1) < 0.9:
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
    return float(np.clip(a, 1, 200)), float(b)
