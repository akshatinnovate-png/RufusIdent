"""Enrolment station backend: a crash-safe queue that turns a short guided
capture (look straight / left / right) into a student's gallery entry.

For each submitted capture:
  1. the raw frames are written to data/enrol_inbox/<job>/ first, so nothing is
     lost if the laptop restarts; unfinished jobs resume on the next start
  2. every frame is checked quickly (one face, big enough, sharp, not turned
     too far) and the best ~20 are picked, spread across the three poses
  3. the picked frames are saved to photos/<admission_no>/ (640 px) and go
     through exactly the same processing as enroll.py (cached, so enroll.bat
     never redoes them)
  4. frames that don't look like the rest (someone else stepped in) are dropped
  5. the face is compared with every other enrolled student: a near-identical
     match means a mistyped admission number or a double enrolment
  6. the student is added to the live gallery straight away; thresholds are
     recalibrated automatically once the queue goes quiet
"""
import datetime as dt
import glob
import json
import os
import re
import shutil
import threading
import time
import uuid

import cv2
import numpy as np

from engine import read_image
from enroll import Cache, process

TARGET = {"straight": 8, "left": 6, "right": 6}
MIN_GOOD = 8           # fewer usable photos than this: ask for a re-take
WARN_BELOW = 14        # enrolled, but a re-take would make recognition stronger
SAME_PERSON = 0.35     # every photo must resemble the student's straight-on photos this much
PHOTO_PX = 640
SAME_FACE = 0.75   # at or above this vs another student: the same person, refuse
MAX_ATTEMPTS = 3   # processing errors are retried; after that the raw frames are parked, never deleted
MIN_FREE_MB = 500  # refuse new captures below this much free disk (the station keeps them meanwhile)
CAPTURE_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class EnrolQueue:
    def __init__(self, cfg, engine, get_gallery, set_gallery):
        self.cfg = cfg
        self.engine = engine
        self.get_gallery = get_gallery
        self.set_gallery = set_gallery
        self.inbox = os.path.join(cfg.data_dir, "enrol_inbox")
        self.parked = os.path.join(cfg.data_dir, "enrol_failed")       # raw frames of jobs that kept erroring
        self.journal = os.path.join(cfg.data_dir, "enrol_journal")     # write-ahead log of enrolled students
        self.meta_path = os.path.join(cfg.data_dir, "enrol_meta.json")
        self.seen_path = os.path.join(cfg.data_dir, "enrol_seen.json")
        for d in (self.inbox, self.parked, self.journal):
            os.makedirs(d, exist_ok=True)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.jobs: dict[str, dict] = {}          # recent jobs for the station screen
        self.meta = self._read_meta()            # admission no -> enrolment record (consent, photos, warnings)
        self.cache = None
        self.dirty_since = None                  # gallery/cache changed and not yet saved
        self.new_since_calibration = 0
        self.calibrating = False
        try:
            with open(self.seen_path, encoding="utf-8") as f:
                self.seen = json.load(f)                 # capture ids already accepted (retries are ignored)
        except (FileNotFoundError, json.JSONDecodeError):
            self.seen = []
        for d in sorted(glob.glob(os.path.join(self.inbox, ".part-*"))):
            shutil.rmtree(d, ignore_errors=True)         # an upload that never finished: the station still has it
        for folder, state in ((self.inbox, "queued"), (self.parked, "failed")):   # resume after a restart
            for d in sorted(glob.glob(os.path.join(folder, "*"))):
                try:
                    with open(os.path.join(d, "job.json"), encoding="utf-8") as f:
                        job = json.load(f)
                    job["state"] = state
                    if state == "failed":
                        job["reprocessable"] = True
                    self.jobs[job["id"]] = job
                except (OSError, json.JSONDecodeError):
                    pass                                 # leave it on disk; never delete frames we can't read

    # ───── records ─────

    def _read_meta(self) -> dict:
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_meta(self):
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)
        os.replace(tmp, self.meta_path)

    def status(self) -> dict:
        g = self.get_gallery()
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["submitted"], reverse=True)[:40]
            queued = sum(1 for j in self.jobs.values() if j["state"] in ("queued", "processing"))
        return {
            "queue": queued,
            "calibrating": self.calibrating,
            "jobs": [{k: v for k, v in j.items() if k != "frames"} for j in jobs],
            "enrolled": {sid: {"photos": int(g.sp_count[i])} for sid, i in g.index.items()} if g else {},
            "students": self.meta,
        }

    # ───── submit ─────

    def submit(self, student: dict, frames: list[tuple[str, bytes]], capture_id: str = "") -> dict:
        """Accept a capture. Only returns once it is safely on disk; the same
        capture sent twice (a retry after a lost reply) is accepted once."""
        sid = student["id"].strip()
        if not sid:
            raise ValueError("Admission number is required.")
        if not frames:
            raise ValueError("No frames were captured.")
        jid = capture_id if CAPTURE_ID.match(capture_id or "") else f"{dt.datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
        with self.lock:
            if jid in self.seen or jid in self.jobs:
                return {"job": jid, "duplicate": True, "queue": self._queued()}
        free_mb = shutil.disk_usage(self.cfg.data_dir).free / 1e6
        if free_mb < MIN_FREE_MB:
            raise OSError(f"Disk almost full ({free_mb:.0f} MB free). Free some space; the station keeps the capture meanwhile.")
        part = os.path.join(self.inbox, f".part-{jid}")
        shutil.rmtree(part, ignore_errors=True)
        os.makedirs(part)
        names = []
        for i, (phase, data) in enumerate(frames):
            phase = phase if phase in TARGET else "straight"
            name = f"{i:03d}_{phase}.jpg"
            _write_durable(os.path.join(part, name), data)
            names.append(name)
        job = {"id": jid, "student": {**student, "id": sid}, "frames": names, "state": "queued",
               "submitted": time.time(), "message": "waiting", "attempts": 0}
        _write_durable(os.path.join(part, "job.json"), json.dumps(job).encode())
        os.replace(part, os.path.join(self.inbox, jid))            # only now is it a real job
        with self.lock:
            self.jobs[jid] = job
            self.seen = (self.seen + [jid])[-20000:]
            _write_durable(self.seen_path, json.dumps(self.seen).encode())
        self.wake.set()
        return {"job": jid, "queue": self._queued()}

    def _queued(self) -> int:
        return sum(1 for j in self.jobs.values() if j["state"] in ("queued", "processing"))

    def reprocess(self, jid: str) -> dict:
        """Put a parked job (its raw frames were kept) back in the queue."""
        src = os.path.join(self.parked, jid)
        if not CAPTURE_ID.match(jid) or not os.path.isdir(src):
            raise ValueError("No parked frames for that job.")
        os.replace(src, os.path.join(self.inbox, jid))
        with self.lock:
            job = self.jobs.get(jid) or json.load(open(os.path.join(self.inbox, jid, "job.json"), encoding="utf-8"))
            job.update(state="queued", attempts=0, message="re-queued", reprocessable=False)
            self.jobs[jid] = job
        self.wake.set()
        return {"job": jid, "queue": self._queued()}

    # ───── write-ahead journal: an enrolled student survives a crash before the next full save ─────

    def _journal_write(self, sid: str, photos: list, best_rel: str):
        counts = np.asarray([len(np.atleast_2d(p)) for p in photos])
        emb = np.concatenate([np.atleast_2d(p) for p in photos]).astype(np.float32)
        path = os.path.join(self.journal, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', sid)}.npz")
        with open(path + ".tmp", "wb") as f:
            np.savez(f, emb=emb, counts=counts, sid=np.array(sid), best=np.array(best_rel or ""))
            f.flush(); os.fsync(f.fileno())
        os.replace(path + ".tmp", path)

    def replay(self, g):
        """Apply journal entries newer than the saved gallery (called at startup and on reload)."""
        files = sorted(glob.glob(os.path.join(self.journal, "*.npz")), key=os.path.getmtime)
        if not files:
            return g
        updates, best = {}, {}
        for path in files:
            try:
                z = np.load(path)
                sid = str(z["sid"])
                parts = np.split(z["emb"], np.cumsum(z["counts"])[:-1])
                updates[sid] = [p for p in parts]
                if str(z["best"]):
                    best[sid] = str(z["best"])
            except Exception:
                continue
        if not updates:
            return g
        from gallery import Gallery
        return Gallery.build(updates, best) if g is None else g.with_students(updates, best, self.cfg.lookalike_pair)

    # ───── worker ─────

    def start(self):
        threading.Thread(target=self._loop, name="enrolment", daemon=True).start()

    def _next(self):
        with self.lock:
            waiting = sorted((j for j in self.jobs.values() if j["state"] == "queued"), key=lambda j: j["submitted"])
            if waiting:
                waiting[0]["state"] = "processing"
                return waiting[0]
        return None

    def _loop(self):
        while True:
            job = self._next()
            if job:
                try:
                    self._process(job)
                except Exception as e:                    # never let one bad job stop the queue
                    self._errored(job, e)
                continue
            self._when_idle()
            self.wake.wait(timeout=20)
            self.wake.clear()

    def _errored(self, job: dict, e: Exception):
        """A processing error (bug, disk, model): retry, then park the raw frames. Never delete them."""
        job["attempts"] = job.get("attempts", 0) + 1
        folder = os.path.join(self.inbox, job["id"])
        try:
            _write_durable(os.path.join(folder, "job.json"), json.dumps({k: v for k, v in job.items()}).encode())
        except OSError:
            pass
        if job["attempts"] < MAX_ATTEMPTS:
            job.update(state="queued", message=f"error, retrying ({job['attempts']}/{MAX_ATTEMPTS}): {e}")
            time.sleep(2)
            return
        try:
            os.replace(folder, os.path.join(self.parked, job["id"]))
        except OSError:
            pass
        job.update(state="failed", reprocessable=True, finished=time.time(),
                   message=f"Processing error after {MAX_ATTEMPTS} tries: {e}. The frames are kept: press REPROCESS.")

    def _save(self):
        if self.dirty_since:
            t0 = time.time()
            g = self.get_gallery()
            if g is not None:
                g.save(self.cfg.data_dir)
                # A journal entry is only deleted once it is in BOTH the new gallery file and the
                # *.prev backup, so falling back to the backup after a bad save can never lose a student.
                stamp = os.path.join(self.journal, "last_save")
                previous = float(open(stamp).read()) if os.path.exists(stamp) else 0.0
                for path in glob.glob(os.path.join(self.journal, "*.npz")):
                    if os.path.getmtime(path) < previous:
                        os.remove(path)
                _write_durable(stamp, str(t0).encode())
            if self.cache is not None:
                self.cache.save()
            self.dirty_since = None

    def _when_idle(self):
        # Save at most once a minute while busy (see _process); always once the queue is empty.
        self._save()
        g = self.get_gallery()
        if g is not None and self.new_since_calibration >= 50 and not self.calibrating:
            self.recalibrate()

    def recalibrate(self):
        g = self.get_gallery()
        if g is None or self.calibrating:
            return
        self.calibrating = True
        try:
            g.calibrate(self.cfg.top_k, self.cfg.lookalike_pair)
            g.save(self.cfg.data_dir)
            self.new_since_calibration = 0
        finally:
            self.calibrating = False

    def _finish(self, job: dict, state: str, message: str, **extra):
        job.update(state=state, message=message, finished=time.time(), **extra)
        shutil.rmtree(os.path.join(self.inbox, job["id"]), ignore_errors=True)
        sid = job["student"]["id"]
        rec = self.meta.get(sid, {})
        rec.update({k: job["student"].get(k) for k in ("name", "cls", "section") if job["student"].get(k)})
        if job["student"].get("consent"):
            rec["consent"] = True
            rec.setdefault("consent_at", dt.datetime.now().isoformat(timespec="seconds"))
        rec.update(last_job=job["id"], last_state=state, last_message=message, updated=dt.datetime.now().isoformat(timespec="seconds"))
        if state == "done":
            rec.update(enrolled_at=rec["updated"], photos=extra.get("photos"), warnings=extra.get("warnings", []), verified=extra.get("verified"))
        self.meta[sid] = rec
        self._write_meta()

    def _select(self, folder: str, names: list[str]) -> list[tuple[str, np.ndarray]]:
        """Quick check of every frame, then the best ones spread across poses and time."""
        good = {p: [] for p in TARGET}
        for order, name in enumerate(names):
            img = read_image(os.path.join(folder, name))
            if img is None:
                continue
            face, _ = self.engine.main_face(img, 320)
            if face is None or face.width < 80 or abs(face.yaw) > self.cfg.max_yaw:
                continue
            if face.blur < self.cfg.min_blur * 1.5:
                continue
            phase = name.split("_", 1)[1].rsplit(".", 1)[0]
            good.setdefault(phase, []).append((order, face.quality, img, name.rsplit(".", 1)[0]))
        picked = []
        spare = []
        for phase, want in TARGET.items():
            frames = sorted(good.get(phase, []))                     # capture order
            if len(frames) <= want:
                picked += frames
                continue
            # evenly spaced through the movement, preferring the sharper of neighbours
            step = len(frames) / want
            chosen = set()
            for k in range(want):
                window = frames[int(k * step):int((k + 1) * step)] or frames[-1:]
                best = max(window, key=lambda f: f[1])
                chosen.add(best[0])
                picked.append(best)
            spare += [f for f in frames if f[0] not in chosen]
        short = sum(TARGET.values()) - len(picked)
        if short > 0:
            picked += sorted(spare, key=lambda f: -f[1])[:short]
        return [(stem, img) for _, _, img, stem in sorted(picked, key=lambda f: f[0])]

    def _process(self, job: dict):
        cfg = self.cfg
        student = job["student"]
        sid = student["id"]
        folder = os.path.join(self.inbox, job["id"])
        job["message"] = "checking frames"
        chosen = self._select(folder, job["frames"])
        if len(chosen) < MIN_GOOD:
            return self._finish(job, "failed", f"Only {len(chosen)} usable frames: please re-take (face the camera, good light, one student only).")

        # Save the picked frames as the student's photos (a re-take replaces the previous station photos).
        job["message"] = "saving photos"
        out_dir = os.path.join(cfg.photos_dir, sid)
        os.makedirs(out_dir, exist_ok=True)
        if self.cache is None:
            self.cache = Cache(cfg.data_dir, fresh=False)
        for old in glob.glob(os.path.join(out_dir, "st_*.jpg")):
            self.cache.items.pop(Cache.key(old), None)
            os.remove(old)
        stamp = f"{dt.datetime.now():%Y%m%d%H%M%S}"
        paths = []
        for tag, img in chosen:
            h, w = img.shape[:2]
            if max(h, w) > PHOTO_PX:
                s = PHOTO_PX / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            path = os.path.join(out_dir, f"st_{stamp}_{tag}.jpg")
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(path)
                paths.append(path)

        # Same processing as enroll.py, results cached for enroll.bat.
        job["message"] = f"processing {len(paths)} photos"
        results = []
        for path in paths:
            r = process(self.engine, cfg, path)
            self.cache.items[Cache.key(path)] = r
            if r["status"] == "ok":
                results.append((path, r))
        if not results:
            return self._finish(job, "failed", "No usable face in the captured frames: please re-take.")

        # Drop anything that isn't the same person as the straight-on photos.
        straight = [r["emb"][0] for p, r in results if "_straight" in os.path.basename(p)] or [r["emb"][0] for _, r in results]
        ref = np.mean(straight, axis=0)
        ref /= np.linalg.norm(ref)
        kept = []
        for p, r in results:
            if float(r["emb"][0] @ ref) >= SAME_PERSON:
                kept.append((p, r))
            else:
                r["status"] = "inconsistent"
        warnings = []
        if len(kept) < MIN_GOOD:
            return self._finish(job, "failed", f"Only {len(kept)} consistent photos: please re-take with only this student in view.")
        if len(kept) < WARN_BELOW:
            warnings.append(f"only {len(kept)} good photos; a re-take would make recognition stronger")

        # Never enrol someone under the wrong number: is this face already another student?
        g = self.get_gallery()
        # Photos added by hand to the same folder stay part of the gallery (processed once if new).
        manual = []
        for p in sorted(glob.glob(os.path.join(out_dir, "*"))):
            if os.path.basename(p).startswith("st_") or os.path.splitext(p)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            k = Cache.key(p)
            if k not in self.cache.items:
                self.cache.items[k] = process(self.engine, cfg, p)
            if self.cache.items[k].get("status") == "ok":
                manual.append(np.atleast_2d(self.cache.items[k]["emb"]))
        if g is not None and len(g.ids):
            for other, score, _ in g.match(ref, cfg.top_k, n=3):
                if other == sid:
                    continue
                if score >= SAME_FACE:
                    # Almost certainly the same person: a mistyped admission number or a double enrolment.
                    for p in paths:
                        self.cache.items.pop(Cache.key(p), None)
                        os.remove(p)
                    return self._finish(job, "failed", f"This face is already enrolled as {other} (match {score:.2f}). Check the admission number; not enrolled.",
                                        duplicate_of=other)
                if score >= g.thresholds(True)["t_accept"]:
                    warnings.append(f"looks a lot like {other} (score {score:.2f}): twins/siblings? The gate will be extra strict between them")
                break

        # Into the live gallery right away.
        job["message"] = "adding to gallery"
        photos = manual + [r["emb"] for _, r in kept]
        best = max(kept, key=lambda x: x[1]["quality"])[0]
        update = {sid: photos}
        best_rel = {sid: os.path.relpath(best, cfg.photos_dir)}
        if g is None:
            from gallery import Gallery
            new = Gallery.build(update, best_rel)
        else:
            new = g.with_students(update, best_rel, cfg.lookalike_pair)
        # Self-check: the student must now be recognised as themselves.
        top = new.match(ref, cfg.top_k, n=1)
        verified = bool(top) and top[0][0] == sid and top[0][1] >= new.thresholds(True)["t_accept"]
        if not verified:
            warnings.append("self-check failed: not reliably recognisable yet — please re-take")
        self._journal_write(sid, photos, best_rel[sid])          # durable before we say "done"
        self.set_gallery(new)
        self.dirty_since = self.dirty_since or time.time()
        self.new_since_calibration += 1
        if time.time() - self.dirty_since > 60:
            self._save()
        self._finish(job, "done", f"{len(kept)} photos enrolled{' · verified ✓' if verified else ''}" + (f" · {warnings[0]}" if warnings else ""),
                     photos=len(kept), warnings=warnings, verified=verified)


def _write_durable(path: str, data: bytes):
    """Write and fsync, so a power cut can't leave a half-written file behind."""
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
