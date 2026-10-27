"""RufusIdent recognition backend.

    python server.py

GET  /health                 status, enrolled students, thresholds
POST /recognize              multipart image (+ track, gate) -> decision for this person
POST /reload                 reload the gallery after running enroll.py
GET  /photo/{admission_no}   the student's best enrolment photo
GET  /student/{admission_no} enrolment details for one student
"""
import datetime as dt
import os
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import load_config
from engine import Engine, decode_image
from fusion import Fusion
from gallery import Gallery

VERSION = "1.0.0"
cfg = load_config()
engine = Engine(cfg)
fusion = Fusion(cfg)
gallery = Gallery.load(cfg.data_dir)
gallery_lock = threading.Lock()

app = FastAPI(title="RufusIdent recognition", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _save_evidence(img, result: dict):
    if not cfg.save_evidence:
        return
    now = dt.datetime.now()
    folder = {"match": "", "review": "_review", "spoof": "_spoof", "unknown": "_unknown"}[result["status"]]
    name = f"{result.get('student_id') or 'unknown'}_{now:%H%M%S}.jpg"
    path = os.path.join(cfg.evidence_dir, f"{now:%Y-%m-%d}", folder, name) if folder else os.path.join(cfg.evidence_dir, f"{now:%Y-%m-%d}", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        buf.tofile(path)


@app.get("/health")
def health():
    g = gallery
    return {
        "ok": True,
        "version": VERSION,
        "students": len(g.ids) if g else 0,
        "photos": int(len(g.emb)) if g else 0,
        "antispoof": engine.spoof is not None,
        "thresholds": None if not g else {"accept": cfg.t_accept or g.t_accept, "review": cfg.t_review or g.t_review, "source": g.calib["source"]},
        "min_frames": cfg.min_frames,
    }


@app.post("/reload")
def reload():
    global gallery
    g = Gallery.load(cfg.data_dir)
    if g is None:
        raise HTTPException(404, "No gallery yet. Run enroll.py first.")
    with gallery_lock:
        gallery = g
    return {"ok": True, "students": len(g.ids)}


@app.post("/recognize")
def recognize(image: UploadFile = File(...), track: str = Form("default"), gate: str = Form("")):
    t0 = time.perf_counter()
    if gallery is None:
        return {"status": "unknown", "reason": "no students enrolled yet", "ms": 0}
    img = decode_image(image.file.read())
    if img is None:
        raise HTTPException(400, "Not an image")

    face, n = engine.main_face(img, cfg.det_size_gate)
    if face is None:
        return {"status": "no_face" if not n else "multiple_faces", "faces": n, "ms": round((time.perf_counter() - t0) * 1000)}
    quality = {"face_px": round(face.width), "sharpness": round(face.blur), "yaw": round(face.yaw, 2), "brightness": round(face.brightness)}
    if face.width < cfg.min_face_px or face.blur < cfg.min_blur:
        return {"status": "low_quality", "quality": quality, "ms": round((time.perf_counter() - t0) * 1000)}

    engine.describe(img, face, tta=False, liveness=True)
    result = fusion.update(track, face.emb, face.quality, face.live, gallery)
    if result.get("decided"):
        _save_evidence(img, result)
    result["quality"] = quality
    result["ms"] = round((time.perf_counter() - t0) * 1000)
    return result


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
    return {"student_id": student_id, "photos": int(gallery.count[i]), "lookalikes": gallery.lookalikes.get(student_id, [])}


if __name__ == "__main__":
    print(f"RufusIdent backend on http://{cfg.host}:{cfg.port}  ·  "
          f"{len(gallery.ids) if gallery else 0} students enrolled  ·  anti-spoof {'on' if engine.spoof else 'off'}")
    if gallery is None:
        print("No gallery yet: put photos in photos/<admission_no>/ and run  python enroll.py")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
