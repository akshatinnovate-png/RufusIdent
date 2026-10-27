"""Measure how fast this laptop runs the recognition pipeline.

    python benchmark.py
"""
import os
import time

import numpy as np

from config import load_config
from engine import Engine


def timed(fn, n=20):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main():
    cfg = load_config()
    e = Engine(cfg)
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    box = np.array([176, 150, 336, 360, 0.9], np.float32)

    det = timed(lambda: e.detector.detect(crop, cfg.det_size_gate, cfg.det_thresh))
    emb = timed(lambda: e.embedder.embed([face], False))
    emb_tta = timed(lambda: e.embedder.embed([face], True), 10)
    spoof = timed(lambda: e.spoof.live_score(crop, box)) if e.spoof else 0.0
    frame = det + emb + spoof
    print(f"CPU threads: {os.cpu_count()}")
    print(f"  detect      {det:6.1f} ms")
    print(f"  embed       {emb:6.1f} ms   (with mirror averaging, used for enrolment: {emb_tta:.1f} ms)")
    print(f"  anti-spoof  {spoof:6.1f} ms")
    print(f"  per frame   {frame:6.1f} ms  → about {1000 / frame:.1f} recognitions per second")
    print(f"  decision after {cfg.min_frames} frames ≈ {frame * cfg.min_frames / 1000:.2f} s per student")
    print(f"  enrolment ≈ {1000 / (timed(lambda: e.detector.detect(crop, cfg.det_size_enroll, cfg.det_thresh), 5) + emb_tta):.1f} photos per second")


if __name__ == "__main__":
    main()
