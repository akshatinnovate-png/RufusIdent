"""RufusIdent recognition backend.

    python server.py

GET  /health                 status, enrolled students, thresholds
POST /frame                  multipart image (a gate camera frame) -> every face in it, tracked and identified
POST /reload                 reload the gallery after running enroll.py
POST /enrol/submit           enrolment station capture (frames + student) -> queued, processed in the background
POST /enrol/identify         is the face at the station already enrolled?
GET  /enrol/status           queue, recent jobs, who is enrolled, consent records
POST /enrol/reprocess        retry a parked job whose frames were kept after repeated errors
POST /enrol/recalibrate      recompute thresholds now
POST /review/confirm         a teacher confirmed a gate "needs review": that face teaches the gallery
GET  /sync/state             what the backend already has (the dashboard pushes the rest)
POST /sync/roster            the student list
POST /sync/day/{date}        one day's attendance
GET  /report/settings        daily email settings (password never returned)
POST /report/settings        change them
GET  /report/status          last sent, next run, errors
POST /report/send            email a report now ({"date": "YYYY-MM-DD", "test": true})
GET  /report/xlsx?date=      download the workbook
GET  /photo/{admission_no}   the student's best enrolment photo
GET  /student/{admission_no} enrolment details for one student
"""
import contextlib
import datetime as dt
import os
import threading
import time

import cv2
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from config import load_config
from enrolment import EnrolQueue
from engine import Engine, decode_image
from fusion import Fusion
from gallery import Gallery
from records import Store
from report import Reporter
from tracker import Tracker

VERSION = "2.0.0"
cfg = load_config()
engine = Engine(cfg)
fusion = Fusion(cfg, liveness=engine.spoof is not None)
tracker = Tracker(ttl_s=3.0)
gallery = Gallery.load(cfg.data_dir)
store = Store(cfg.data_dir)
reporter = Reporter(cfg.data_dir, os.path.join(os.path.dirname(cfg.evidence_dir), "reports"), store)
frame_lock = threading.Lock()          # one frame at a time keeps tracking consistent
state = {"frames": 0, "learned": 0, "last_adapt_save": time.monotonic(), "verifying_near": False}
review_faces: dict[str, object] = {}   # track -> best face of a "needs review" decision


def _set_gallery(g):
    global gallery
    with frame_lock:
        gallery = g


enrol = EnrolQueue(cfg, engine, lambda: gallery, _set_gallery)
gallery = enrol.replay(gallery)         # students enrolled after the last full save (write-ahead journal)


@contextlib.asynccontextmanager
async def lifespan(_app):
    reporter.start()
    enrol.start()
    yield
    if gallery is not None and len(gallery.adapt):
        gallery.save_adaptive(cfg.data_dir)      # keep what was learned at the gate


app = FastAPI(title="RufusIdent recognition", version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _save_evidence(img, box, result: dict):
    if not cfg.save_evidence:
        return
    now = dt.datetime.now()
    x1, y1, x2, y2 = box[:4]
    side = max(x2 - x1, y2 - y1) * 3
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = img.shape[:2]
    crop = img[int(max(0, cy - side / 2)):int(min(h, cy + side / 2)), int(max(0, cx - side / 2)):int(min(w, cx + side / 2))]
    sub = {"match": "", "review": "_review", "spoof": "_spoof", "unknown": "_unknown"}[result["status"]]
    folder = os.path.join(cfg.evidence_dir, f"{now:%Y-%m-%d}", sub)
    os.makedirs(folder, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        buf.tofile(os.path.join(folder, f"{result.get('student_id') or 'unknown'}_{now:%H%M%S_%f}"[:-3] + ".jpg"))


@app.get("/health")
def health():
    g = gallery
    return {
        "ok": True,
        "version": VERSION,
        "students": len(g.ids) if g else 0,
        "photos": int(len(g.photo_start)) if g else 0,
        "learned_samples": int(len(g.adapt)) if g else 0,
        "antispoof": engine.spoof is not None,
        "device": engine.device,
        "calibration": None if not g else g.calib,
    }


@app.post("/reload")
def reload():
    global gallery
    g = enrol.replay(Gallery.load(cfg.data_dir))
    if g is None:
        raise HTTPException(404, "No gallery yet (or made by an older version). Run enroll.py.")
    with frame_lock:
        gallery = g
    return {"ok": True, "students": len(g.ids)}


@app.post("/frame")
def frame(image: UploadFile = File(...), gate: str = Form("")):
    t0 = time.perf_counter()
    img = decode_image(image.file.read())
    if img is None:
        raise HTTPException(400, "Not an image")
    with frame_lock:
        state["frames"] += 1
        # The far scan is the most expensive step: skip it while someone close is
        # being verified, so their answer comes back sooner.
        far_scan = state["frames"] % cfg.far_scan_every == 1 and not state["verifying_near"]
        faces = engine.scene(img, far_scan, tracker.small_boxes(cfg.near_px))
        tids = tracker.assign([f.box for f in faces])

        out, todo = [], []
        for f, tid in zip(faces, tids):
            item = {"track": tid, "box": [round(float(v), 1) for v in f.box[:4]], "width": round(f.width), "det": round(f.det, 3)}
            done = fusion.cached(tid)
            if gallery is None:
                item.update(status="unknown", reason="no students enrolled yet")
            elif done:
                item.update({k: v for k, v in done.items() if k != "candidates"}, cached=True)
            elif f.width < cfg.recog_min_px:
                item.update(status="too_far")
            elif f.blur < cfg.min_blur * min(1.0, f.width / 112) or abs(f.yaw) > 0.6:
                item.update(status="low_quality", reason="blurred" if abs(f.yaw) <= 0.6 else "turned away")
            else:
                todo.append((f, item))
            out.append(item)

        state["verifying_near"] = any(f.width >= cfg.near_px for f, _ in todo)
        engine.describe_many(img, [f for f, _ in todo], cfg.live_min_px)
        for f, item in todo:
            r = fusion.update(item["track"], f.emb, f.quality, f.width, f.live, gallery)
            learn = r.pop("_learn", None)
            if learn is not None and r["student_id"] in gallery.index:
                gallery.add_adaptive(r["student_id"], learn, cfg.adaptive_per_student)
                state["learned"] += 1
            best = r.pop("_best", None)
            if best is not None:
                review_faces[item["track"]] = best
                while len(review_faces) > 300:
                    review_faces.pop(next(iter(review_faces)))
            if r.get("decided"):
                _save_evidence(img, f.box, r)
            item.update(r)

        if state["learned"] and time.monotonic() - state["last_adapt_save"] > 60:
            gallery.save_adaptive(cfg.data_dir)
            state["learned"], state["last_adapt_save"] = 0, time.monotonic()

    return {"faces": out, "far_scan": far_scan, "ms": round((time.perf_counter() - t0) * 1000)}


# ───────────── attendance sync + daily report ─────────────

@app.get("/sync/state")
def sync_state():
    return store.sync_state()


@app.post("/sync/roster")
def sync_roster(body: dict = Body(...)):
    store.put_roster(body.get("roster", []), bool(body.get("demo")), body.get("hash", ""))
    return {"ok": True}


@app.post("/sync/day/{date}")
def sync_day(date: str, body: dict = Body(...)):
    try:
        store.put_day(date, body.get("records", {}), body.get("hash", ""))
    except ValueError:
        raise HTTPException(400, "Date must be YYYY-MM-DD")
    return {"ok": True}


@app.get("/report/settings")
def report_settings():
    return reporter.public_settings()


@app.post("/report/settings")
def report_settings_update(body: dict = Body(...)):
    return reporter.update_settings(body)


@app.get("/report/status")
def report_status():
    return reporter.status()


@app.post("/report/send")
def report_send(body: dict = Body(default={})):
    try:
        return reporter.run(body.get("date"), test=bool(body.get("test")))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.get("/report/xlsx")
def report_xlsx(date: str = ""):
    date = date or dt.date.today().isoformat()
    try:
        xlsx = reporter.build(date)[0]
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(xlsx, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="GHS_Attendance_{date}.xlsx"'})


# ───────────── enrolment station ─────────────

@app.post("/enrol/submit")
def enrol_submit(frames: list[UploadFile] = File(...), student_id: str = Form(...), name: str = Form(""),
                 cls: str = Form(""), section: str = Form(""), consent: str = Form(""), capture_id: str = Form("")):
    data = [((f.filename or "").split("_", 1)[-1].rsplit(".", 1)[0], f.file.read()) for f in frames]
    try:
        return enrol.submit({"id": student_id, "name": name, "cls": cls, "section": section, "consent": consent in ("1", "true", "yes")},
                            data, capture_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(507, str(e))          # disk problem: the station keeps the capture and retries


@app.post("/enrol/reprocess")
def enrol_reprocess(body: dict = Body(...)):
    try:
        return enrol.reprocess(str(body.get("job", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/enrol/identify")
def enrol_identify(image: UploadFile = File(...)):
    """Is the student now standing at the station already enrolled?"""
    img = decode_image(image.file.read())
    g = gallery
    if img is None or g is None or not len(g.ids):
        return {"known": False}
    face, _ = engine.main_face(img, 320)
    if face is None:
        return {"known": False}
    engine.describe(img, face, tta=False, liveness=False)
    best = g.match(face.emb, cfg.top_k, n=1)
    if best and best[0][1] >= g.thresholds(True)["t_accept"]:
        return {"known": True, "student_id": best[0][0], "score": round(best[0][1], 3)}
    return {"known": False}


@app.get("/enrol/status")
def enrol_status():
    return enrol.status()


@app.post("/enrol/recalibrate")
def enrol_recalibrate():
    if gallery is None:
        raise HTTPException(400, "No students enrolled yet.")
    threading.Thread(target=enrol.recalibrate, daemon=True).start()
    return {"ok": True, "message": "Recalibrating in the background (about a minute for a full school)."}


@app.post("/review/confirm")
def review_confirm(body: dict = Body(...)):
    """A teacher confirmed a 'needs review' student at the gate: remember that face."""
    emb = review_faces.pop(str(body.get("track", "")), None)
    sid = str(body.get("student_id", ""))
    if emb is None or gallery is None or sid not in gallery.index:
        return {"ok": True, "learned": False}
    with frame_lock:
        gallery.add_adaptive(sid, emb, cfg.adaptive_per_student)
        gallery.save_adaptive(cfg.data_dir)
    return {"ok": True, "learned": True}


@app.get("/photo/{student_id}")
def photo(student_id: str):
    rel = gallery.best_photo.get(student_id) if gallery else None
    if not rel:
        raise HTTPException(404, "No photo")
    path = os.path.normpath(os.path.join(cfg.photos_dir, rel))
    if not path.startswith(os.path.normpath(cfg.photos_dir)) or not os.path.exists(path):
        raise HTTPException(404, "No photo")
    return FileResponse(path, headers={"Cache-Control": "max-age=3600"})


@app.get("/student/{student_id}")
def student(student_id: str):
    if not gallery or student_id not in gallery.index:
        raise HTTPException(404, "Not enrolled")
    i = gallery.index[student_id]
    return {"student_id": student_id, "photos": int(gallery.sp_count[i]), "lookalikes": gallery.lookalikes.get(student_id, []),
            "learned_samples": int((gallery.adapt_owner == i).sum())}


def _port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":      # same as uvicorn: a port in TIME_WAIT after a restart is free (Windows has no such state)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    if not _port_free(cfg.host, cfg.port):
        print(f"\nPort {cfg.port} is already in use by another program (or another copy of this backend).\n"
              f"  See what is using it:   netstat -ano | findstr :{cfg.port}\n"
              f"  Or pick another port:   put {{\"port\": 8898}} in backend\\config.json and set\n"
              f"                          Settings → Gate → backend URL to http://127.0.0.1:8898")
        raise SystemExit(1)
    print(f"RufusIdent backend {VERSION} on http://{cfg.host}:{cfg.port}  ·  {engine.device}  ·  "
          f"{len(gallery.ids) if gallery else 0} students enrolled  ·  anti-spoof {'on' if engine.spoof else 'off'}")
    if gallery is None:
        print("No gallery yet: put photos in photos/<admission_no>/ and run  python enroll.py")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
