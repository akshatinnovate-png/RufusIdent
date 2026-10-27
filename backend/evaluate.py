"""How accurate is recognition at each distance, on YOUR students' photos?

    python evaluate.py                          # 1080p webcam, 78° field of view
    python evaluate.py --width 3840 --fov 40    # e.g. a 4K camera zoomed in
    python evaluate.py --students 300 --frames 5

Run enroll.py first. For a sample of students, one photo is held out of the
gallery. That photo is then shrunk to the size the student's face would be at
1-6 m through the given camera, with webcam noise, blur and JPEG added, and
put through the same detect -> embed -> multi-frame decision as the gate.
Liveness is not part of this test.
"""
import argparse
import math
import os
import zlib

import cv2
import numpy as np

from config import load_config
from engine import Engine, read_image
from enroll import Cache, scan
from gallery import Gallery

FACE_M = 0.15


def gate_view(img: np.ndarray, face_w: float, target_px: float, rng, out_w: int) -> np.ndarray:
    s = target_px / face_w
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR).astype(np.float32)
    small = np.clip(small + rng.normal(0, 4, small.shape), 0, 255).astype(np.uint8)
    small = cv2.GaussianBlur(small, (3, 3), 0.7)
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], cv2.IMREAD_COLOR)
    out_h = out_w * 9 // 16
    frame = np.full((out_h, out_w, 3), 80, np.uint8)
    h, w = min(small.shape[0], out_h), min(small.shape[1], out_w)
    y, x = (out_h - h) // 2, (out_w - w) // 2
    frame[y:y + h, x:x + w] = small[:h, :w]
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1920, help="camera horizontal resolution")
    ap.add_argument("--fov", type=float, default=78, help="camera horizontal field of view in degrees")
    ap.add_argument("--students", type=int, default=100)
    ap.add_argument("--frames", type=int, default=5, help="frames fused per decision")
    ap.add_argument("--distances", default="1,2,3,4,5,6")
    args = ap.parse_args()

    cfg = load_config()
    engine = Engine(cfg)
    cache = Cache(cfg.data_dir, fresh=False)
    if not cache.items:
        raise SystemExit("Run enroll.py first.")
    by_path = {k.split("|")[1]: v for k, v in cache.items.items()}
    # Use enrolment's final verdicts: photos it rejected or flagged as misfiled are left out.
    import csv
    with open(os.path.join(cfg.data_dir, "enroll_report.csv"), encoding="utf-8") as f:
        usable = {os.path.abspath(os.path.join(cfg.photos_dir, r["photo"])) for r in csv.DictReader(f)
                  if r["status"] == "ok" and not r["warning"]}

    per_student, held = {}, {}
    for sid, files in scan(cfg.photos_dir).items():
        good = [(p, by_path.get(os.path.abspath(p))) for p in files]
        good = [(p, m) for p, m in good if m and m["status"] == "ok" and os.path.abspath(p) in usable]
        if not good:
            continue
        if len(good) >= 5 and len(held) < args.students:
            held[sid] = good[-1][0]
            good = good[:-1]
        per_student[sid] = [m["emb"] for _, m in good]
    if not held:
        raise SystemExit("Need students with at least 5 good photos.")
    gallery = Gallery.build(per_student, {})
    print("Calibrating on the reduced gallery…")
    gallery.calibrate(cfg.top_k, cfg.lookalike_pair)

    dists = [float(d) for d in args.distances.split(",")]
    px_at = {d: FACE_M * args.width / (2 * d * math.tan(math.radians(args.fov) / 2)) for d in dists}
    print(f"\nCamera {args.width}px wide, {args.fov:.0f}° field of view · {len(held)} students · {args.frames} frames each\n")
    print(f"{'distance':>8} {'face':>6} {'found':>7} {'top-1':>7} {'auto ✓':>8} {'WRONG':>7} {'review':>7} {'unsure':>7}")

    for d in dists:
        tally = dict(found=0, top1=0, auto=0, wrong=0, review=0, unsure=0)
        mistakes = set()
        for sid, path in held.items():
            img = read_image(path)
            f0, _ = engine.main_face(img, cfg.det_size_enroll)
            if f0 is None:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{path}{d}".encode()))
            embs, weights, widths = [], [], []
            for _ in range(args.frames):
                frame = gate_view(img, f0.width, px_at[d], rng, args.width)
                faces = engine.scene(frame, far_scan=True, small_hints=[])
                if not faces:
                    continue
                f = faces[0]
                if f.width < cfg.recog_min_px:
                    continue
                engine.describe_many(frame, [f], live_min_px=10**6)
                embs.append(f.emb); weights.append(f.quality); widths.append(f.width)
            if not embs:
                continue
            tally["found"] += 1
            fused = np.average(np.stack(embs), axis=0, weights=weights)
            fused /= np.linalg.norm(fused)
            cands = gallery.match(fused, cfg.top_k, n=2)
            best, s1, base = cands[0]
            s2 = cands[1][1] if len(cands) > 1 else 0
            th = gallery.thresholds(np.average(widths, weights=weights) >= cfg.near_px)
            tally["top1"] += best == sid
            lookalike = len(cands) > 1 and gallery.is_lookalike(best, cands[1][0])
            sure = s1 >= th["t_accept"] and base >= th["t_review"] and s1 - s2 >= cfg.margin * (2 if lookalike else 1)
            if sure:
                tally["auto" if best == sid else "wrong"] += 1
                if best != sid:
                    mistakes.add(f"{sid} marked as {best} (score {s1:.2f})")
            elif s1 >= th["t_review"]:
                tally["review"] += 1
            else:
                tally["unsure"] += 1
        n = len(held)
        pc = lambda k: f"{tally[k] / n * 100:6.1f}%"
        print(f"{d:7.1f}m {px_at[d]:5.0f}px {pc('found')} {pc('top1')} {pc('auto'):>8} {pc('wrong')} {pc('review')} {pc('unsure')}")
        for m in sorted(mistakes):
            print(f"          ! {m}")
    print("\nfound = face detected · top-1 = right student ranked first · auto ✓ = marked automatically and correctly\n"
          "WRONG = marked automatically as the wrong student · review = sent to the teacher · unsure = not marked")


if __name__ == "__main__":
    main()
