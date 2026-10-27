"""Settings for the recognition backend. Override any of them in config.json."""
import json
import os
from dataclasses import asdict, dataclass, field, fields

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    photos_dir: str = os.path.join(HERE, "..", "photos")   # photos/<admission_no>/*.jpg
    roster_file: str = ""                                    # optional .xlsx/.csv, for the enrolment report
    data_dir: str = os.path.join(HERE, "data")
    models_dir: str = os.path.join(HERE, "models")
    evidence_dir: str = os.path.join(HERE, "evidence")      # point this at the OneDrive folder
    host: str = "127.0.0.1"
    port: int = 8000
    threads: int = 0                                         # 0 = all CPU cores
    provider: str = "auto"          # auto | cpu | directml (Windows integrated graphics)

    # Detection
    det_size_enroll: int = 640
    det_size_gate: int = 640        # every frame: faces down to ~30 px
    det_size_far: int = 1280        # far scan: faces down to ~18 px
    far_scan_every: int = 4         # frames between far scans
    det_thresh: float = 0.5
    min_face_px: int = 60           # enrolment photos
    recog_min_px: int = 24          # gate: smaller faces are tracked but not identified yet
    near_px: int = 64               # gate faces at least this wide use the 'near' thresholds
    live_min_px: int = 48           # anti-spoof needs at least this many pixels
    min_blur: float = 12.0          # Laplacian variance on the aligned 112px face
    max_yaw: float = 0.55           # enrolment; the gate screen already asks for frontal faces

    # Recognition
    tta: bool = True                # average the embedding of the face and its mirror image
    far_variants: list = field(default_factory=lambda: [28, 48])   # enrolment also stores each photo as seen from far (face px)
    top_k: int = 3                  # student score = mean of their best k photo matches
    t_accept: float = 0.0           # 0 = use the calibrated value from enrolment
    t_review: float = 0.0
    margin: float = 0.08            # best student must beat the runner-up by this much
    lookalike_pair: float = 0.55    # centroid similarity that flags two students as look-alikes

    # Multi-frame fusion at the gate
    min_frames: int = 3             # near faces
    min_frames_far: int = 5         # far faces: more frames averaged before deciding
    max_frames: int = 16
    switch_guard: float = 0.30      # a frame this unlike the track so far = a different person
    vote_share: float = 0.6
    track_ttl_s: float = 4.0
    retry_after_s: float = 3.0

    # Anti-spoofing (printed photo / phone screen)
    antispoof: bool = True
    live_threshold: float = 0.55
    require_live: bool = True       # never auto-mark until liveness was checked at least once

    # Learn each student's look under the gate camera from very confident matches
    adaptive: bool = True
    adaptive_per_student: int = 5

    save_evidence: bool = True


def load_config(path: str = os.path.join(HERE, "config.json")) -> Config:
    cfg = Config()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        known = {f.name for f in fields(Config)}
        for k, v in user.items():
            if k not in known:
                raise ValueError(f"config.json: unknown setting '{k}'")
            setattr(cfg, k, v)
    for k in ("photos_dir", "data_dir", "models_dir", "evidence_dir", "roster_file"):
        v = getattr(cfg, k)
        if v and not os.path.isabs(v):
            setattr(cfg, k, os.path.normpath(os.path.join(HERE, v)))
    return cfg


def dump_defaults() -> str:
    return json.dumps(asdict(Config()), indent=2)
