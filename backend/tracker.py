"""Keeps the same id on the same person across frames, for several people at once."""
import itertools
import time

import numpy as np


class Tracker:
    def __init__(self, ttl_s: float = 1.5):
        self.ttl = ttl_s
        self.tracks: dict[str, dict] = {}       # id -> {box, last}
        self._ids = itertools.count(1)

    def assign(self, boxes: list[np.ndarray]) -> list[str]:
        """Track id for each box. Boxes are matched to the nearest existing track
        whose last box is close (relative to face size) and of similar size."""
        now = time.monotonic()
        for k in [k for k, t in self.tracks.items() if now - t["last"] > self.ttl]:
            del self.tracks[k]
        pairs = []
        for i, b in enumerate(boxes):
            bw, cx, cy = b[2] - b[0], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for tid, t in self.tracks.items():
                tb = t["box"]
                tw = tb[2] - tb[0]
                ratio = bw / max(tw, 1)
                dist = np.hypot(cx - (tb[0] + tb[2]) / 2, cy - (tb[1] + tb[3]) / 2) / max(bw, tw, 1)
                if 0.6 < ratio < 1.7 and dist < 1.2:
                    pairs.append((dist, i, tid))
        out: list[str | None] = [None] * len(boxes)
        used = set()
        for dist, i, tid in sorted(pairs):
            if out[i] is None and tid not in used:
                out[i] = tid
                used.add(tid)
        for i, b in enumerate(boxes):
            if out[i] is None:
                out[i] = f"t{next(self._ids)}"
            self.tracks[out[i]] = {"box": np.asarray(b[:4], np.float32), "last": now}
        return out

    def small_boxes(self, max_width: float) -> list[np.ndarray]:
        return [t["box"] for t in self.tracks.values() if t["box"][2] - t["box"][0] < max_width]
