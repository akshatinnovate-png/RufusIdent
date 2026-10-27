"""Face pipeline: detect (SCRFD-10G) -> align (5-point similarity) ->
embed (ArcFace R50, WebFace600K) + quality + anti-spoofing (MiniFASNet x2).

Everything runs on the CPU through ONNX Runtime, so it works on a laptop
without a graphics card.
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from config import Config

# Where ArcFace expects the eyes, nose and mouth corners in a 112x112 crop.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

MODEL_FILES = {
    "detector": "det_10g.onnx",
    "embedder": "w600k_r50.onnx",
    "spoof_v2": "MiniFASNetV2.onnx",
    "spoof_v1se": "MiniFASNetV1SE.onnx",
}


PROVIDERS = ["CPUExecutionProvider"]


def _session(path: str, threads: int, providers: list[str] | None = None, quiet: bool = False) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads   # otherwise ONNX Runtime uses one thread per physical core (fastest on laptops)
    opts.log_severity_level = 4 if quiet else 3
    return ort.InferenceSession(path, opts, providers=providers or PROVIDERS)


def pick_providers(cfg: Config, embedder_path: str) -> tuple[list[str], str]:
    """Use the laptop's built-in graphics through DirectML (Windows) when it is
    available, gives the same answers as the CPU, and is actually faster."""
    if cfg.provider == "cpu" or "DmlExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"], "CPU"
    import time
    face = np.random.default_rng(0).integers(0, 255, (4, 3, 112, 112)).astype(np.float32) / 127.5 - 1
    results = {}
    for name, prov in (("CPU", ["CPUExecutionProvider"]), ("DirectML", ["DmlExecutionProvider", "CPUExecutionProvider"])):
        try:
            sess = _session(embedder_path, cfg.threads, prov, quiet=True)
            inp = sess.get_inputs()[0].name
            out = sess.run(None, {inp: face})[0]
            t = time.perf_counter()
            for _ in range(5):
                sess.run(None, {inp: face})
            results[name] = (time.perf_counter() - t, out / np.linalg.norm(out, axis=1, keepdims=True), prov)
        except Exception:
            pass
    if "DirectML" in results and "CPU" in results:
        same = float(np.min(np.sum(results["DirectML"][1] * results["CPU"][1], axis=1)))
        if same > 0.995 and (cfg.provider == "directml" or results["DirectML"][0] < results["CPU"][0] * 0.9):
            return results["DirectML"][2], "DirectML (integrated graphics)"
    return ["CPUExecutionProvider"], "CPU"


def degrade(face112: np.ndarray, px: int, rng: np.random.Generator) -> np.ndarray:
    """What this face would look like about `px` pixels wide through a webcam:
    downsampled, sensor noise, JPEG, then back up to 112px."""
    r = max(8, int(round(112 * px / 100)))
    small = cv2.resize(face112, (r, r), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = np.clip(small + rng.normal(0, 3, small.shape), 0, 255).astype(np.uint8)
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)
    return cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)


def read_image(path: str) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def decode_image(buf: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


# ───────────────────────── detection ─────────────────────────

class SCRFD:
    STRIDES = (8, 16, 32)
    ANCHORS = 2

    def __init__(self, path: str, threads: int, providers=None):
        self.sess = _session(path, threads, providers)
        self.input = self.sess.get_inputs()[0].name
        self._centers: dict = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self._centers[key] = np.repeat(c, self.ANCHORS, axis=0)
        return self._centers[key]

    def detect(self, img: np.ndarray, size: int, thresh: float):
        """Returns (boxes Nx5 [x1,y1,x2,y2,score], kps Nx5x2) in image pixels."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input: blob})

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.STRIDES):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= thresh)[0]
            if not keep.size:
                continue
            centers = self._anchor_centers(size // stride, size // stride, stride)[keep]
            d = outs[i + 3].reshape(-1, 4)[keep] * stride
            k = outs[i + 6].reshape(-1, 10)[keep] * stride
            boxes.append(np.stack([centers[:, 0] - d[:, 0], centers[:, 1] - d[:, 1], centers[:, 0] + d[:, 2], centers[:, 1] + d[:, 3]], axis=1))
            kpss.append((k.reshape(-1, 5, 2) + centers[:, None, :]))
            scores.append(sc[keep])
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        keep = _nms(boxes, scores, 0.4)
        return np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32), kpss[keep].astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= iou]
    return keep


def align(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
    return cv2.warpAffine(img, m, (112, 112), borderValue=0)


# ───────────────────────── embedding ─────────────────────────

class ArcFace:
    def __init__(self, path: str, threads: int, providers=None):
        self.sess = _session(path, threads, providers)
        self.input = self.sess.get_inputs()[0].name

    def embed(self, faces: list[np.ndarray], tta: bool) -> tuple[np.ndarray, np.ndarray]:
        """Unit-length 512-d embeddings plus the raw feature norm (a quality signal)."""
        batch = faces + [cv2.flip(f, 1) for f in faces] if tta else faces
        blob = cv2.dnn.blobFromImages(batch, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        feats = self.sess.run(None, {self.input: blob})[0]
        if tta:
            n = len(faces)
            feats = feats[:n] + feats[n:]
        norms = np.linalg.norm(feats, axis=1)
        return feats / norms[:, None], norms / (2 if tta else 1)


# ───────────────────────── anti-spoofing ─────────────────────────

class AntiSpoof:
    """MiniFASNetV2 (2.7x context) + MiniFASNetV1SE (4.0x context), averaged.
    Class 1 of the 3 outputs is 'real face'."""

    def __init__(self, v2: str, v1se: str, threads: int):
        self.models = [(_session(v2, threads), 2.7), (_session(v1se, threads), 4.0)]

    @staticmethod
    def _crop(img: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box[:4]
        bw, bh = x2 - x1, y2 - y1
        s = min((h - 1) / bh, (w - 1) / bw, scale)
        cx, cy = x1 + bw / 2, y1 + bh / 2
        nx1, ny1 = max(0, int(cx - bw * s / 2)), max(0, int(cy - bh * s / 2))
        nx2, ny2 = min(w - 1, int(cx + bw * s / 2)), min(h - 1, int(cy + bh * s / 2))
        return cv2.resize(img[ny1:ny2 + 1, nx1:nx2 + 1], (80, 80))

    def live_score(self, img: np.ndarray, box: np.ndarray) -> float:
        probs = []
        for sess, scale in self.models:
            x = self._crop(img, box, scale).astype(np.float32).transpose(2, 0, 1)[None]
            logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
            e = np.exp(logits - logits.max())
            probs.append(float(e[1] / e.sum()))
        return float(np.mean(probs))


# ───────────────────────── quality ─────────────────────────

def pose(kps: np.ndarray) -> tuple[float, float]:
    """Rough yaw (left/right) and pitch (up/down) from the 5 landmarks."""
    le, re, nose, lm, rm = kps
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    eye_dist = np.linalg.norm(re - le) + 1e-6
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / (mouth_mid[1] - eye_mid[1] + 1e-6) - 0.55)
    return yaw, pitch


def blur(face112: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(face112, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


@dataclass
class Face:
    box: np.ndarray
    kps: np.ndarray
    det: float
    width: float
    yaw: float
    pitch: float
    blur: float
    brightness: float
    aligned: np.ndarray = field(repr=False)
    emb: np.ndarray | None = field(default=None, repr=False)
    norm: float = 0.0
    live: float | None = None

    @property
    def quality(self) -> float:
        """0..1 weight used when fusing several frames. Bigger, sharper, more
        frontal faces count for more."""
        sharp_ref = 60 * min(1.0, self.width / 112)
        q = min(1.0, self.width / 140) * min(1.0, self.blur / max(sharp_ref, 5)) * max(0.0, 1 - abs(self.yaw) * 1.4)
        q *= min(1.0, self.det / 0.8) * min(1.0, self.norm / 20) if self.norm else 1.0
        return float(max(q, 0.05))


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        p = lambda k: os.path.join(cfg.models_dir, MODEL_FILES[k])
        missing = [MODEL_FILES[k] for k in ("detector", "embedder") if not os.path.exists(p(k))]
        if missing:
            raise FileNotFoundError(f"Missing models in {cfg.models_dir}: {', '.join(missing)}. Run: python setup_models.py")
        providers, self.device = pick_providers(cfg, p("embedder"))
        self.detector = SCRFD(p("detector"), cfg.threads, providers)
        self.embedder = ArcFace(p("embedder"), cfg.threads, providers)
        self.spoof = None
        if cfg.antispoof and os.path.exists(p("spoof_v2")) and os.path.exists(p("spoof_v1se")):
            self.spoof = AntiSpoof(p("spoof_v2"), p("spoof_v1se"), cfg.threads)

    def faces(self, img: np.ndarray, det_size: int) -> list[Face]:
        boxes, kpss = self.detector.detect(img, det_size, self.cfg.det_thresh)
        return self._faces(img, boxes, kpss)

    def scene(self, img: np.ndarray, far_scan: bool, small_hints: list) -> list[Face]:
        """Every face in a gate frame, near or far.
        1. the whole frame at the normal size (faces down to ~30 px)
        2. on far-scan frames, the whole frame at double size (down to ~18 px)
        3. a zoomed look around each small face already being tracked, so far
           students stay tracked every frame without paying for step 2"""
        cfg = self.cfg
        h, w = img.shape[:2]
        parts = [self.detector.detect(img, cfg.det_size_gate, cfg.det_thresh)]
        if far_scan:
            parts.append(self.detector.detect(img, cfg.det_size_far, cfg.det_thresh))
        for x1, y1, x2, y2 in small_hints:
            side = max(x2 - x1, y2 - y1) * 4
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ox, oy = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            crop = img[oy:int(min(h, cy + side / 2)), ox:int(min(w, cx + side / 2))]
            if crop.size == 0:
                continue
            b, k = self.detector.detect(crop, 320, cfg.det_thresh)
            if len(b):
                b[:, [0, 2]] += ox; b[:, [1, 3]] += oy
                k[:, :, 0] += ox; k[:, :, 1] += oy
            parts.append((b, k))
        boxes = np.concatenate([b for b, _ in parts])
        kpss = np.concatenate([k for _, k in parts])
        if len(boxes) > 1:
            keep = _nms(boxes[:, :4], boxes[:, 4], 0.4)
            boxes, kpss = boxes[keep], kpss[keep]
        return self._faces(img, boxes, kpss)

    def _faces(self, img: np.ndarray, boxes: np.ndarray, kpss: np.ndarray) -> list[Face]:
        out = []
        for b, k in zip(boxes, kpss):
            al = align(img, k)
            yaw, pitch = pose(k)
            out.append(Face(box=b, kps=k, det=float(b[4]), width=float(b[2] - b[0]), yaw=yaw, pitch=pitch,
                            blur=blur(al), brightness=float(al.mean()), aligned=al))
        out.sort(key=lambda f: f.width, reverse=True)
        return out

    def main_face(self, img: np.ndarray, det_size: int) -> tuple[Face | None, int]:
        """The biggest face, provided it clearly dominates (else None = ambiguous)."""
        faces = self.faces(img, det_size)
        if not faces:
            return None, 0
        if len(faces) > 1 and faces[1].width > faces[0].width * 0.8:
            return None, len(faces)
        return faces[0], len(faces)

    def describe_many(self, img: np.ndarray, faces: list[Face], live_min_px: int) -> None:
        """Embed several faces in one batch; liveness only where the face is big enough to judge."""
        if not faces:
            return
        embs, norms = self.embedder.embed([f.aligned for f in faces], tta=False)
        for f, e, n in zip(faces, embs, norms):
            f.emb, f.norm = e, float(n)
            if self.spoof is not None and f.width >= live_min_px:
                f.live = self.spoof.live_score(img, f.box)

    def describe(self, img: np.ndarray, face: Face, tta: bool, liveness: bool) -> Face:
        embs, norms = self.embedder.embed([face.aligned], tta)
        face.emb, face.norm = embs[0], float(norms[0])
        if liveness and self.spoof is not None:
            face.live = self.spoof.live_score(img, face.box)
        return face
