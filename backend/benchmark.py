"""Measure how fast this laptop runs the gate pipeline.

    python benchmark.py
"""
import os
import time

import numpy as np

from config import load_config
from engine import Engine


def timed(fn, n=10):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main():
    cfg = load_config()
    e = Engine(cfg)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1080, 1080, 3), dtype=np.uint8)       # the centre-square gate lane of a 1080p camera
    faces = [rng.integers(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
    box = np.array([400, 300, 560, 510, 0.9], np.float32)

    det = timed(lambda: e.detector.detect(frame, cfg.det_size_gate, cfg.det_thresh))
    far = timed(lambda: e.detector.detect(frame, cfg.det_size_far, cfg.det_thresh), 3)
    emb1 = timed(lambda: e.embedder.embed(faces[:1], False))
    emb4 = timed(lambda: e.embedder.embed(faces, False), 5)
    spoof = timed(lambda: e.spoof.live_score(frame, box)) if e.spoof else 0.0
    tta = timed(lambda: e.embedder.embed(faces[:1], True), 5)

    one = det + emb1 + spoof
    four = det + emb4 + 4 * spoof
    avg_det = det + far / cfg.far_scan_every
    print(f"Device: {e.device} · CPU threads: {os.cpu_count()}")
    print(f"  detect (every frame)        {det:7.1f} ms")
    print(f"  far scan (1 in {cfg.far_scan_every} frames)     {far:7.1f} ms")
    print(f"  identify 1 face             {emb1 + spoof:7.1f} ms")
    print(f"  identify 4 faces at once    {emb4 + 4 * spoof:7.1f} ms")
    print(f"  → 1 student in view:  about {1000 / (avg_det + emb1 + spoof):4.1f} frames/s, decision in ~{cfg.min_frames * (avg_det + emb1 + spoof) / 1000:.1f} s")
    print(f"  → 4 students in view: about {1000 / (avg_det + emb4 + 4 * spoof):4.1f} frames/s, all four decided in ~{cfg.min_frames * (avg_det + emb4 + 4 * spoof) / 1000:.1f} s")
    print(f"  → once a student is marked, their face is no longer re-checked (only detection runs)")
    print(f"  enrolment ≈ {1000 / (timed(lambda: e.detector.detect(frame, cfg.det_size_enroll, cfg.det_thresh), 3) + tta + emb1 * len(cfg.far_variants)):.1f} photos/s")


if __name__ == "__main__":
    main()
