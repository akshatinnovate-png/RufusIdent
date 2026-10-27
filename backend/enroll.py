"""Enrol every student from photos/<admission_no>/*.jpg.

    python enroll.py            add new/changed photos (finished photos are cached)
    python enroll.py --rebuild  start from scratch

Writes data/gallery.*, data/enroll_report.csv and prints what needs fixing:
bad photos, photos that look like a different student, students with too
few usable photos, and look-alike students.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import zlib

import numpy as np

from config import load_config
from engine import Engine, degrade, read_image
from gallery import Gallery

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_GOOD = 8            # warn below this many usable photos per student
CONSISTENCY = 0.30      # a photo must resemble the student's other photos at least this much
ID_ALIASES = {"admission_no", "admission no", "adm no", "adm_no", "admno", "admission number", "id", "student id", "student_id"}


def scan(photos_dir: str) -> dict[str, list[str]]:
    out = {}
    if not os.path.isdir(photos_dir):
        sys.exit(f"Photos folder not found: {photos_dir}\nCreate photos/<admission_no>/ folders with each student's photos.")
    for sid in sorted(os.listdir(photos_dir)):
        d = os.path.join(photos_dir, sid)
        if os.path.isdir(d):
            files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in EXTS]
            if files:
                out[sid.strip()] = files
    return out


def roster_ids(path: str) -> set[str] | None:
    if not path or not os.path.exists(path):
        return None
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    else:
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True).active
        rows = [["" if v is None else str(v) for v in r] for r in ws.iter_rows(values_only=True)]
    head = [h.strip().lower() for h in rows[0]]
    col = next((i for i, h in enumerate(head) if h in ID_ALIASES), None)
    if col is None:
        sys.exit(f"{path}: no admission number column")
    return {r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()}


class Cache:
    """Per-photo results keyed by path + size + mtime, so re-runs only process new photos."""

    def __init__(self, data_dir: str, fresh: bool):
        self.path = os.path.join(data_dir, "embed_cache.npz")
        self.items: dict[str, dict] = {}
        if not fresh and os.path.exists(self.path):
            z = np.load(self.path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            for i, (k, m) in enumerate(meta.items()):
                row, nv = m.pop("row"), m.pop("nv", 1)
                m["emb"] = z["emb"][row:row + nv].astype(np.float32) if row is not None else None
                self.items[k] = m

    @staticmethod
    def key(path: str) -> str:
        st = os.stat(path)
        return f"v2|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"

    def save(self):
        embs, meta = [], {}
        for k, m in self.items.items():
            mm = {kk: v for kk, v in m.items() if kk != "emb"}
            if m.get("emb") is not None:
                mm["row"], mm["nv"] = sum(len(e) for e in embs), len(m["emb"])
                embs.append(m["emb"])
            else:
                mm["row"] = None
            meta[k] = mm
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, emb=(np.concatenate(embs) if embs else np.zeros((0, 512))).astype(np.float16), meta=json.dumps(meta))
        os.replace(tmp, self.path)


def process(engine: Engine, cfg, path: str) -> dict:
    img = read_image(path)
    if img is None:
        return {"status": "unreadable"}
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        import cv2
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    face, n = engine.main_face(img, cfg.det_size_enroll)
    if face is None:
        return {"status": "multiple_faces" if n else "no_face", "faces": n}
    info = {"width": round(face.width), "blur": round(face.blur, 1), "yaw": round(face.yaw, 2), "det": round(face.det, 3)}
    if face.width < cfg.min_face_px:
        return {"status": "too_small", **info}
    if face.blur < cfg.min_blur:
        return {"status": "blurry", **info}
    if abs(face.yaw) > cfg.max_yaw:
        return {"status": "turned_away", **info}
    engine.describe(img, face, tta=True, liveness=False)
    # The same face as the gate camera would see it from far away.
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    far = [degrade(face.aligned, px, rng) for px in cfg.far_variants]
    far_embs = engine.embedder.embed(far, tta=False)[0] if far else np.zeros((0, 512), np.float32)
    emb = np.vstack([face.emb[None], far_embs]).astype(np.float32)
    return {"status": "ok", "quality": round(face.quality, 3), "norm": round(face.norm, 2), **info, "emb": emb}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and process every photo again")
    args = ap.parse_args()

    cfg = load_config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    students = scan(cfg.photos_dir)
    total = sum(len(v) for v in students.values())
    print(f"Found {len(students)} student folders, {total} photos in {cfg.photos_dir}")

    engine = Engine(cfg)
    cache = Cache(cfg.data_dir, args.rebuild)
    results: dict[str, list[tuple[str, dict]]] = {}
    todo = [(sid, p) for sid, files in students.items() for p in files]
    pending = sum(1 for _, p in todo if Cache.key(p) not in cache.items)
    if pending:
        print(f"{pending} new or changed photos to process ({total - pending} already done)")
    t0, done_new = time.time(), 0
    for i, (sid, p) in enumerate(todo, 1):
        k = Cache.key(p)
        if k not in cache.items:
            cache.items[k] = process(engine, cfg, p)
            done_new += 1
            if done_new % 300 == 0:
                cache.save()
            if done_new % 50 == 0 or done_new == pending:
                rate = done_new / (time.time() - t0)
                print(f"  {done_new}/{pending} photos · {rate:.1f}/s · about {(pending - done_new) / max(rate, 1e-6) / 60:.0f} min left", flush=True)
        results.setdefault(sid, []).append((p, cache.items[k]))
    cache.save()

    # Per student: keep photos that agree with each other.
    per_student, best_photo, rows = {}, {}, []
    for sid, items in results.items():
        ok = [(p, m) for p, m in items if m["status"] == "ok"]
        if len(ok) >= 3:
            E = np.stack([m["emb"][0] for _, m in ok])
            S = E @ E.T
            mean_other = (S.sum(1) - 1) / (len(ok) - 1)
            for (p, m), v in zip(ok, mean_other):
                m["consistency"] = round(float(v), 3)
                if v < CONSISTENCY:
                    m["status"] = "inconsistent"
        good = [(p, m) for p, m in items if m["status"] == "ok"]
        if good:
            per_student[sid] = [m["emb"] for _, m in good]
            best_photo[sid] = os.path.relpath(max(good, key=lambda x: x[1]["quality"])[0], cfg.photos_dir)
        for p, m in items:
            rows.append([sid, os.path.relpath(p, cfg.photos_dir), m["status"], m.get("quality", ""), m.get("width", ""), m.get("blur", ""), m.get("yaw", ""), m.get("consistency", ""), ""])

    if not per_student:
        sys.exit("No usable photos found. Check data/enroll_report.csv for the reasons.")

    gallery = Gallery.build(per_student, best_photo)

    # Photos that look more like another student than their own folder (misfiled photos).
    misfiled = 0
    row_of = {(r[0], r[1]): r for r in rows}
    for sid, items in results.items():
        if sid not in gallery.index:
            continue
        own = gallery.index[sid]
        for p, m in items:
            if m["status"] != "ok":
                continue
            q = m["emb"][0]
            others = gallery.match(q, k=cfg.top_k, n=3)
            other = next(((o, s) for o, s, _ in others if o != sid), None)
            seg = np.asarray([float(gallery.emb[gallery.photo_start[ph]] @ q)
                              for ph in range(gallery.sp_start[own], gallery.sp_start[own] + gallery.sp_count[own])])
            seg = np.sort(seg)[::-1][1:1 + cfg.top_k]          # leave this photo out
            mine = float(seg.mean()) if seg.size else 1.0
            if other and other[1] > mine and other[1] >= 0.35:
                row_of[(sid, os.path.relpath(p, cfg.photos_dir))][8] = f"looks like {other[0]} ({other[1]:.2f} vs own {mine:.2f})"
                misfiled += 1

    print("Calibrating thresholds from the enrolment photos…")
    report = gallery.calibrate(cfg.top_k, cfg.lookalike_pair)
    gallery.save(cfg.data_dir)

    with open(os.path.join(cfg.data_dir, "enroll_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["admission_no", "photo", "status", "quality", "face_px", "sharpness", "yaw", "consistency", "warning"])
        w.writerows(rows)

    status_count: dict[str, int] = {}
    for r in rows:
        status_count[r[2]] = status_count.get(r[2], 0) + 1
    few = sorted(sid for sid in students if len(per_student.get(sid, [])) < MIN_GOOD)
    roster = roster_ids(cfg.roster_file)
    summary = {
        "students_enrolled": len(per_student),
        "photos_total": total,
        "photos_by_status": status_count,
        "misfiled_suspects": misfiled,
        "students_below_min_photos": few,
        "calibration": report,
    }
    if roster is not None:
        summary["roster_without_photos"] = sorted(roster - set(per_student))
        summary["photo_folders_not_in_roster"] = sorted(set(students) - roster)
    with open(os.path.join(cfg.data_dir, "enroll_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"\nEnrolled {len(per_student)} students from {status_count.get('ok', 0)} good photos.")
    for st, n in sorted(status_count.items()):
        if st != "ok":
            print(f"  {n:6d} photos rejected: {st}")
    if misfiled:
        print(f"  {misfiled:6d} photos look like a DIFFERENT student (see 'warning' column)")
    if few:
        print(f"  {len(few):6d} students have fewer than {MIN_GOOD} usable photos: {', '.join(few[:15])}{' …' if len(few) > 15 else ''}")
    if roster is not None and summary["roster_without_photos"]:
        print(f"  {len(summary['roster_without_photos']):6d} roster students have no photos")
    print(f"  Look-alike students: {report['lookalike_students']}")
    print(f"  Thresholds: {report['source']}")
    for level, label in (("near", "near faces (64 px+)"), ("far", "far faces")):
        r = report[level]
        line = f"    {label:20s} accept ≥ {r['t_accept']:.3f}, review ≥ {r['t_review']:.3f}"
        if r["genuine_accept_rate"] is not None:
            line += f" · self-test: {r['genuine_accept_rate'] * 100:.2f}% genuine pass, {(r['impostor_pass_rate'] or 0) * 100:.3f}% impostors pass"
        print(line)
    if "warning" in report:
        print(f"  WARNING: {report['warning']}")
    print(f"Report: {os.path.join(cfg.data_dir, 'enroll_report.csv')}")

    try:
        urllib.request.urlopen(urllib.request.Request(f"http://{cfg.host}:{cfg.port}/reload", method="POST"), timeout=3)
        print("Running backend reloaded the new gallery.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
