import base64
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from iris.io.dataclasses import IRImage, IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher
from iris.pipelines.iris_pipeline import IRISPipeline

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe import Image as MPImage
    import urllib.request

    _MP_AVAILABLE = True
except Exception as e:
    mp = None
    _MP_AVAILABLE = False
    _mp_import_error = str(e)


app = FastAPI()

logging.basicConfig(
    level=os.environ.get("OPEN_IRIS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("open_iris_service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception("%s %s -> exception (%.1fms)", request.method, request.url.path, duration_ms)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


EyeSide = Literal["left", "right", "both"]


class EnrollFaceRequest(BaseModel):
    person_id: str
    person_name: str
    image: str  # base64 encoded jpeg/png
    eye_side: EyeSide = "both"
    face_box: Optional[Tuple[float, float, float, float]] = Field(default=None, description="x,y,w,h")
    store_images: bool = False


class IdentifyFaceRequest(BaseModel):
    image: str  # base64 encoded jpeg/png
    eye_side: EyeSide = "both"
    face_box: Optional[Tuple[float, float, float, float]] = Field(default=None, description="x,y,w,h")
    threshold: float = 0.4


IRIS_DB_FILE = os.environ.get("OPEN_IRIS_DB_FILE", "iris_templates.json")
OPEN_IRIS_IMAGE_DIR = os.environ.get("OPEN_IRIS_IMAGE_DIR", "iris_images")


# Load pipeline with config that uses min_iris_diameter=50 (default is 150 which is too strict)
_CONFIG_PATH = os.path.join(_SRC_DIR, "iris", "pipelines", "confs", "pipeline.yaml")
iris_pipeline = IRISPipeline(config=_CONFIG_PATH)
matcher = HammingDistanceMatcher(rotation_shift=15, normalise=True, norm_mean=0.45, separate_half_matching=True)

_FACE_LANDMARKER = None
_mp_error = None
_FACE_LANDMARKER_MODEL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

if _MP_AVAILABLE:
    try:
        # Download model if not exists
        if not os.path.exists(_FACE_LANDMARKER_MODEL):
            print("[iris_service] Downloading FaceLandmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, _FACE_LANDMARKER_MODEL)
            print("[iris_service] Model downloaded")
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_LANDMARKER_MODEL),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=False,
            output_face_blendshapes=False,
        )
        _FACE_LANDMARKER = FaceLandmarker.create_from_options(options)
        print("[iris_service] MediaPipe FaceLandmarker initialized successfully")
    except Exception as e:
        _FACE_LANDMARKER = None
        _mp_error = str(e)
        print(f"[iris_service] MediaPipe FaceLandmarker initialization failed: {e}")
else:
    print(f"[iris_service] MediaPipe not available: {_mp_import_error if not _MP_AVAILABLE else ''}")

# Log the status at startup
print(f"[iris_service] Eye detection methods: mediapipe={_FACE_LANDMARKER is not None}, haar=available")


def _load_db() -> Dict[str, Any]:
    if os.path.exists(IRIS_DB_FILE):
        with open(IRIS_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(db: Dict[str, Any]) -> None:
    with open(IRIS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def _decode_base64_image(b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _encode_base64_image(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face_box")
    return x1, y1, x2, y2


def _crop_face(img_bgr: np.ndarray, face_box: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if face_box is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(face_box, width=w, height=h)
    return img_bgr[y1:y2, x1:x2]


def _crop_from_landmarks(img_bgr: np.ndarray, landmarks_xy: List[Tuple[float, float]], pad_ratio: float = 0.35) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    xs = [p[0] for p in landmarks_xy]
    ys = [p[1] for p in landmarks_xy]
    x1 = int(max(0, min(xs) * w))
    y1 = int(max(0, min(ys) * h))
    x2 = int(min(w, max(xs) * w))
    y2 = int(min(h, max(ys) * h))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img_bgr.dtype)
    return img_bgr[y1:y2, x1:x2]


def _detect_eyes_mediapipe(face_bgr: np.ndarray) -> Dict[Literal["left", "right"], np.ndarray]:
    if _FACE_LANDMARKER is None:
        print("[iris_service] MediaPipe not initialized, skipping")
        return {}

    # Convert BGR to RGB and create MPImage
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_img = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    
    result = _FACE_LANDMARKER.detect(mp_img)
    if result.face_landmarks is None or len(result.face_landmarks) == 0:
        print("[iris_service] MediaPipe found no face landmarks in face crop")
        return {}

    print(f"[iris_service] MediaPipe found {len(result.face_landmarks)} faces")

    # Get iris landmarks (indices 468-479 for refined iris landmarks)
    lm = result.face_landmarks[0]
    left_idx = list(range(468, 474))
    right_idx = list(range(474, 480))
    h, w = face_bgr.shape[:2]
    left_pts = [(lm[i].x * w, lm[i].y * h) for i in left_idx if i < len(lm)]
    right_pts = [(lm[i].x * w, lm[i].y * h) for i in right_idx if i < len(lm)]

    eyes: Dict[Literal["left", "right"], np.ndarray] = {}
    if len(left_pts) > 0:
        crop = _crop_from_landmarks(face_bgr, left_pts, pad_ratio=0.55)
        if crop.size > 0:
            eyes["left"] = crop
    if len(right_pts) > 0:
        crop = _crop_from_landmarks(face_bgr, right_pts, pad_ratio=0.55)
        if crop.size > 0:
            eyes["right"] = crop
    if len(eyes) > 0:
        print(f"[iris_service] eye_detection=mediapipe sides={sorted(list(eyes.keys()))}")
    return eyes


def _detect_eyes_with_method(face_bgr: np.ndarray) -> Tuple[Dict[Literal["left", "right"], np.ndarray], str]:
    eyes = _detect_eyes_mediapipe(face_bgr)
    if len(eyes) > 0:
        return eyes, "mediapipe"

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_candidates = [
        os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml"),
        os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"),
    ]

    h, w = gray.shape[:2]
    upper_h = max(1, int(h * 0.65))
    upper = gray[:upper_h, :]

    detections: Optional[np.ndarray] = None
    for cascade_path in cascade_candidates:
        eye_cascade = cv2.CascadeClassifier(cascade_path)
        if eye_cascade.empty():
            continue

        passes = [
            {"scaleFactor": 1.1, "minNeighbors": 3},
            {"scaleFactor": 1.05, "minNeighbors": 2},
            {"scaleFactor": 1.2, "minNeighbors": 2},
        ]
        for p in passes:
            det = eye_cascade.detectMultiScale(
                upper,
                scaleFactor=p["scaleFactor"],
                minNeighbors=p["minNeighbors"],
                minSize=(max(10, int(w * 0.06)), max(10, int(h * 0.06))),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if det is not None and len(det) > 0:
                detections = det
                break
        if detections is not None:
            break

    if detections is not None and len(detections) > 0:
        dets = sorted(detections.tolist(), key=lambda b: b[2] * b[3], reverse=True)[:2]
        dets = sorted(dets, key=lambda b: b[0])
        eyes_out: Dict[Literal["left", "right"], np.ndarray] = {}
        for idx, (x, y, ew, eh) in enumerate(dets):
            pad_x = int(round(ew * 0.5))  # Increased from 0.35 for better iris segmentation
            pad_y = int(round(eh * 0.5))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(upper.shape[1], x + ew + pad_x)
            y2 = min(upper.shape[0], y + eh + pad_y)
            eye_bgr = face_bgr[y1:y2, x1:x2]
            if eye_bgr.size == 0:
                continue
            side: Literal["left", "right"] = "left" if idx == 0 else "right"
            eyes_out[side] = eye_bgr
        if len(eyes_out) > 0:
            return eyes_out, "haar"

    # Fallback split - larger area for better iris capture
    y1 = int(round(h * 0.08))  # Expanded from 0.12
    y2 = int(round(h * 0.58))  # Expanded from 0.55
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    left_x1 = int(round(w * 0.05))
    left_x2 = int(round(w * 0.48))
    right_x1 = int(round(w * 0.52))
    right_x2 = int(round(w * 0.95))
    eyes_out = {}
    left_eye = face_bgr[y1:y2, left_x1:left_x2]
    right_eye = face_bgr[y1:y2, right_x1:right_x2]
    if left_eye.size > 0:
        eyes_out["left"] = left_eye
    if right_eye.size > 0:
        eyes_out["right"] = right_eye
    return eyes_out, "fallback_split"


def _detect_eyes(face_bgr: np.ndarray) -> Dict[Literal["left", "right"], np.ndarray]:
    eyes, method = _detect_eyes_with_method(face_bgr)
    logger.debug("eye_detection=%s sides=%s", method, sorted(list(eyes.keys())))
    return eyes


@app.post("/debug_extract_eyes")
async def debug_extract_eyes(request: IdentifyFaceRequest) -> Dict[str, Any]:
    try:
        img_bgr = _decode_base64_image(request.image)
        face_bgr = _crop_face(img_bgr, request.face_box)
        eyes, method = _detect_eyes_with_method(face_bgr)
        out: Dict[str, Any] = {"success": True, "method": method, "sides": sorted(list(eyes.keys()))}
        if "left" in eyes:
            out["left_image"] = _encode_base64_image(eyes["left"])
        if "right" in eyes:
            out["right_image"] = _encode_base64_image(eyes["right"])
        return out
    except Exception as e:
        return {"success": False, "error": str(e)}


class DebugSegmentRequest(BaseModel):
    image: str  # base64 encoded eye image (not face, just eye crop)
    eye_side: Literal["left", "right"] = "left"


@app.post("/debug_segment")
async def debug_segment(request: DebugSegmentRequest) -> Dict[str, Any]:
    """Debug endpoint to check if an eye image can be properly segmented."""
    try:
        eye_bgr = _decode_base64_image(request.image)
        if eye_bgr.size == 0:
            return {"success": False, "error": "Empty eye image"}
        
        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        
        # Run just the segmentation part
        result = iris_pipeline.estimate(IRImage(img_data=gray, image_id="debug", eye_side=request.eye_side))
        
        if isinstance(result, dict) and result.get("error") is not None:
            error_info = result["error"]
            return {
                "success": False, 
                "error": error_info if isinstance(error_info, str) else error_info.get("message", str(error_info)),
                "error_type": error_info.get("error_type") if isinstance(error_info, dict) else None,
            }
        
        # Extract some debug info
        debug_info = {"success": True}
        if isinstance(result, dict):
            if "geometry_polygons" in result:
                gp = result["geometry_polygons"]
                debug_info["iris_diameter"] = float(gp.iris_diameter) if hasattr(gp, "iris_diameter") else None
                debug_info["pupil_diameter"] = float(gp.pupil_diameter) if hasattr(gp, "pupil_diameter") else None
            if "eye_orientation" in result:
                debug_info["eye_orientation"] = float(result["eye_orientation"].angle) if hasattr(result["eye_orientation"], "angle") else None
        
        return debug_info
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _build_template(eye_bgr: np.ndarray, image_id: str, eye_side: Literal["left", "right"]) -> IrisTemplate:
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    result = iris_pipeline.estimate(IRImage(img_data=gray, image_id=image_id, eye_side=eye_side))
    if isinstance(result, dict) and result.get("error") is not None:
        error_info = result["error"]
        # Parse error to provide more helpful message
        if isinstance(error_info, dict):
            error_type = error_info.get("error_type", "")
            error_msg = error_info.get("message", str(error_info))
            if "VectorizationError" in error_type or "contours" in error_msg.lower():
                raise RuntimeError(f"Iris segmentation failed - eye image quality too poor or no clear iris detected. Error: {error_msg}")
            raise RuntimeError(f"{error_type}: {error_msg}")
        raise RuntimeError(str(error_info))
    template = result["iris_template"] if isinstance(result, dict) else getattr(result, "iris_template", None)
    if template is None:
        raise RuntimeError("No iris template generated")
    return template


def _ensure_person(db: Dict[str, Any], person_id: str, person_name: str) -> None:
    if person_id not in db:
        db[person_id] = {"name": person_name, "templates": {"left": [], "right": []}}
    if "templates" not in db[person_id]:
        db[person_id]["templates"] = {"left": [], "right": []}
    if "left" not in db[person_id]["templates"]:
        db[person_id]["templates"]["left"] = []
    if "right" not in db[person_id]["templates"]:
        db[person_id]["templates"]["right"] = []
    if person_name and db[person_id].get("name") != person_name:
        db[person_id]["name"] = person_name


def _store_images(person_id: str, face_bgr: np.ndarray, eyes: Dict[str, np.ndarray]) -> None:
    os.makedirs(OPEN_IRIS_IMAGE_DIR, exist_ok=True)
    person_dir = os.path.join(OPEN_IRIS_IMAGE_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)
    ts = str(int(cv2.getTickCount()))
    cv2.imwrite(os.path.join(person_dir, f"face_{ts}.jpg"), face_bgr)
    for side, eye in eyes.items():
        cv2.imwrite(os.path.join(person_dir, f"eye_{side}_{ts}.jpg"), eye)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "endpoints": ["/health", "/enroll_face", "/identify_face", "/database", "/docs"]}


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        logger.info("health_check")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/enroll_face")
async def enroll_face(request: EnrollFaceRequest) -> Dict[str, Any]:
    try:
        logger.info(
            "enroll_face person_id=%s eye_side=%s face_box=%s store_images=%s",
            request.person_id,
            request.eye_side,
            request.face_box,
            request.store_images,
        )
        img_bgr = _decode_base64_image(request.image)
        face_bgr = _crop_face(img_bgr, request.face_box)
        eyes = _detect_eyes(face_bgr)

        wanted: List[Literal["left", "right"]]
        if request.eye_side == "both":
            wanted = ["left", "right"]
        else:
            wanted = [request.eye_side]

        db = _load_db()
        _ensure_person(db, request.person_id, request.person_name)

        enrolled: List[str] = []
        for side in wanted:
            if side not in eyes:
                continue
            template = _build_template(eyes[side], image_id=request.person_id, eye_side=side)
            db[request.person_id]["templates"][side].append(template.serialize())
            enrolled.append(side)

        logger.info("enroll_face person_id=%s enrolled_sides=%s", request.person_id, enrolled)

        _save_db(db)

        if request.store_images:
            _store_images(request.person_id, face_bgr, eyes)

        template_count = len(db[request.person_id]["templates"]["left"]) + len(db[request.person_id]["templates"]["right"])
        if len(enrolled) == 0:
            return {"success": False, "error": "No eyes detected/enrolled", "person_id": request.person_id}

        return {
            "success": True,
            "person_id": request.person_id,
            "enrolled": enrolled,
            "template_count": template_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/identify_face")
async def identify_face(request: IdentifyFaceRequest) -> Dict[str, Any]:
    try:
        logger.info("identify_face eye_side=%s face_box=%s threshold=%s", request.eye_side, request.face_box, request.threshold)
        img_bgr = _decode_base64_image(request.image)
        face_bgr = _crop_face(img_bgr, request.face_box)
        eyes = _detect_eyes(face_bgr)

        wanted: List[Literal["left", "right"]]
        if request.eye_side == "both":
            wanted = ["left", "right"]
        else:
            wanted = [request.eye_side]

        probe_templates: Dict[Literal["left", "right"], IrisTemplate] = {}
        for side in wanted:
            if side in eyes:
                probe_templates[side] = _build_template(eyes[side], image_id="probe", eye_side=side)

        if len(probe_templates) == 0:
            return {"success": False, "error": "No eyes detected", "matched": False}

        logger.info("identify_face probe_sides=%s", sorted(list(probe_templates.keys())))

        db = _load_db()
        best_match: Optional[Dict[str, Any]] = None
        best_score = 1.0

        for person_id, person_data in db.items():
            templates_by_side = person_data.get("templates", {})
            for side, probe_template in probe_templates.items():
                for stored in templates_by_side.get(side, []) or []:
                    try:
                        gallery_template = IrisTemplate.deserialize(stored)
                    except Exception:
                        continue
                    score = matcher.run(probe_template, gallery_template)
                    if score < best_score:
                        best_score = float(score)
                        best_match = {"person_id": person_id, "name": person_data.get("name"), "eye_side": side}

        if best_match is not None and best_score < request.threshold:
            logger.info(
                "identify_face matched person_id=%s score=%.4f eye_side=%s",
                best_match["person_id"],
                best_score,
                best_match.get("eye_side"),
            )
            return {
                "success": True,
                "matched": True,
                "person_id": best_match["person_id"],
                "name": best_match.get("name"),
                "score": best_score,
                "eye_side": best_match.get("eye_side"),
            }

        logger.info("identify_face no_match best_score=%s threshold=%s", best_score if best_match is not None else None, request.threshold)
        return {
            "success": True,
            "matched": False,
            "best_score": best_score if best_match is not None else None,
            "threshold": request.threshold,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "matched": False}


@app.get("/database")
async def get_database() -> Dict[str, Any]:
    db = _load_db()
    # Return metadata only (no full templates)
    out: Dict[str, Any] = {}
    for person_id, person_data in db.items():
        templates = person_data.get("templates", {})
        out[person_id] = {
            "name": person_data.get("name"),
            "left_count": len(templates.get("left", []) or []),
            "right_count": len(templates.get("right", []) or []),
        }
    return out


@app.delete("/database/{person_id}")
async def delete_person(person_id: str) -> Dict[str, Any]:
    db = _load_db()
    if person_id in db:
        del db[person_id]
        _save_db(db)
        return {"success": True, "deleted": person_id}
    return {"success": False, "error": "person_id not found"}


@app.post("/recognize")
async def recognize_compat(request: IdentifyFaceRequest) -> Dict[str, Any]:
    return await identify_face(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)