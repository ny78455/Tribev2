"""
aese/adapters/character_stub.py
Character presence adapter — deterministic OpenCV-only path.

V4 (Fix 1 — 2026-08-18): Removed FastVLM (generative) primary path.
Character counting is a classification/detection task, not a generation task.
The VLM path was causing max_characters_seen=0 across all events because
count_people() returned filler text that the regex parser could not parse as
an integer. See DECISIONS.md §16.

Detector chain (in priority order):
  1. OpenCV FaceDetectorYN / YuNet ONNX (OpenCV 5+)
  2. OpenCV DNN SSD ResNet10 (OpenCV 4, Caffe model files present)
  3. OpenCV Haar cascade frontal + profile (OpenCV 4, bundled XMLs)
  4. 0 — last resort (logs a warning; see _init_detector)

No character identity, no tracking, no names, no re-identification.
This is an intentional stub per the contract (§1.2, §5.0).
See DECISIONS.md §4, §16.

Future work: replace with a proper face-tracking + re-ID pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect OpenCV major version once
# ---------------------------------------------------------------------------
_OCV_VERSION_MAJOR = int(cv2.__version__.split(".")[0])

# ---------------------------------------------------------------------------
# Detector singletons — set on first _init_detector() call
# ---------------------------------------------------------------------------
_face_net = None          # cv2.dnn.Net (DNN SSD, OpenCV 4 only)
_yunet = None             # cv2.FaceDetectorYN (OpenCV 5+)
_haar_frontal = None      # cv2.CascadeClassifier (OpenCV 4 only)
_haar_profile = None      # cv2.CascadeClassifier (OpenCV 4 only)

# "yunet" | "dnn" | "opencv_haar_frontal+profile" | "opencv_haar_frontal" | "none"
_detector_mode: str = "none"
_detector_init_done = False

# Confidence threshold for DNN SSD detector (OpenCV 4 path)
_DNN_CONF_THRESHOLD = 0.5
# Confidence threshold for YuNet (score is in the last column of each detection row)
_YUNET_CONF_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Paths to local model files
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "models"))

# YuNet ONNX model (OpenCV 5 FaceDetectorYN)
_YUNET_MODEL = os.path.join(_MODELS_DIR, "yunet", "face_detection_yunet_2023mar.onnx")

# DNN SSD ResNet10 Caffe model (OpenCV 4 only)
_DNN_PROTO_CANDIDATES = [
    os.path.join(_MODELS_DIR, "deploy.prototxt"),
    "/usr/share/opencv4/haarcascades/deploy.prototxt",
]
_DNN_MODEL_CANDIDATES = [
    os.path.join(_MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
]

# Haar cascade XMLs — cv2.data.haarcascades first, then local repo copy
# (handles opencv-python-headless installs where cv2/data/ is empty)
_LOCAL_HAARCASCADES = os.path.join(_MODELS_DIR, "haarcascades")


def _find_cascade(filename: str) -> Optional[str]:
    """Return the path to a Haar cascade XML, searching cv2.data first then local repo."""
    candidates = [
        cv2.data.haarcascades + filename,
        os.path.join(_LOCAL_HAARCASCADES, filename),
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_detector() -> None:
    """
    Initialize the best available face detector.
    Priority: YuNet (OCV5) > DNN SSD (OCV4) > Haar (OCV4) > none.
    """
    global _face_net, _yunet, _haar_frontal, _haar_profile
    global _detector_mode, _detector_init_done
    if _detector_init_done:
        return
    _detector_init_done = True

    # ---- Path A: OpenCV 5+ — FaceDetectorYN (YuNet ONNX) ----------------
    if _OCV_VERSION_MAJOR >= 5 and hasattr(cv2, "FaceDetectorYN"):
        if os.path.isfile(_YUNET_MODEL):
            try:
                # Input size is set dynamically per-frame in _opencv_count_characters
                _yunet = cv2.FaceDetectorYN.create(_YUNET_MODEL, "", (320, 320))
                _detector_mode = "yunet"
                logger.info("AESE character_stub: YuNet face detector loaded (OpenCV 5+).")
                return
            except Exception as exc:
                logger.debug("YuNet face detector load failed: %s", exc)
        else:
            logger.warning(
                "AESE character_stub: OpenCV 5 detected but YuNet model not found at %s. "
                "Run: download face_detection_yunet_2023mar.onnx to models/yunet/",
                _YUNET_MODEL,
            )

    # ---- Path B: OpenCV 4 — DNN SSD ResNet10 (Caffe) --------------------
    if _OCV_VERSION_MAJOR < 5:
        proto = next((p for p in _DNN_PROTO_CANDIDATES if os.path.isfile(p)), None)
        model = next((m for m in _DNN_MODEL_CANDIDATES if os.path.isfile(m)), None)
        if proto and model:
            try:
                _face_net = cv2.dnn.readNet(proto, model)
                _detector_mode = "dnn"
                logger.info("AESE character_stub: DNN SSD face detector loaded (OpenCV 4).")
                return
            except Exception as exc:
                logger.debug("DNN SSD face detector load failed: %s", exc)

    # ---- Path C: OpenCV 4 — Haar cascade (frontal + profile) ------------
    if _OCV_VERSION_MAJOR < 5 and hasattr(cv2, "CascadeClassifier"):
        frontal_path = _find_cascade("haarcascade_frontalface_default.xml")
        profile_path = _find_cascade("haarcascade_profileface.xml")

        if frontal_path:
            _haar_frontal = cv2.CascadeClassifier(frontal_path)
            if profile_path:
                _haar_profile = cv2.CascadeClassifier(profile_path)
                _detector_mode = "opencv_haar_frontal+profile"
                logger.info(
                    "AESE character_stub: Haar cascade (frontal+profile) loaded (OpenCV 4)."
                )
            else:
                _detector_mode = "opencv_haar_frontal"
                logger.info(
                    "AESE character_stub: Haar cascade (frontal only) loaded (OpenCV 4)."
                )
            return

    logger.warning(
        "AESE character_stub: No face detector available — count_characters will always return 0. "
        "For OpenCV 5+: ensure models/yunet/face_detection_yunet_2023mar.onnx exists. "
        "This is a STUB; see DECISIONS.md §4."
    )
    _detector_mode = "none"


# ---------------------------------------------------------------------------
# Public: effective detector chain name
# ---------------------------------------------------------------------------

def get_effective_detector_chain() -> str:
    """
    Return a short string describing the active face detection chain.

    Possible values (FastVLM is intentionally NOT a valid return value —
    character counting must never route through a generative model):
      "yunet"                     — OpenCV FaceDetectorYN (YuNet ONNX, OpenCV 5+)
      "dnn"                       — OpenCV DNN SSD ResNet10 (OpenCV 4)
      "opencv_haar_frontal+profile" — Haar frontal + profile cascades (OpenCV 4)
      "opencv_haar_frontal"       — Haar frontal cascade only (OpenCV 4)
      "none"                      — No detector available

    Triggers detector initialisation if not yet done.
    """
    # Always use the deterministic OpenCV chain — never FastVLM for character counting.
    # See DECISIONS.md §16 for why the generative path was removed.
    _init_detector()
    return _detector_mode


# ---------------------------------------------------------------------------
# IoU deduplication helper (used by Haar path)
# ---------------------------------------------------------------------------

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union of two rectangles (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedup_and_count(
    frontal: List[Tuple[int, int, int, int]],
    profile: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> int:
    """
    Crude deduplication: merge frontal + profile detections, suppress duplicates
    with IoU > iou_threshold. Returns the count of unique detections.
    """
    all_boxes: List[Tuple[int, int, int, int]] = list(frontal) + list(profile)
    if not all_boxes:
        return 0

    kept: List[Tuple[int, int, int, int]] = []
    for box in all_boxes:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return len(kept)


# ---------------------------------------------------------------------------
# Public: count_characters
# ---------------------------------------------------------------------------

def count_characters(image: Optional[np.ndarray]) -> int:
    """
    Count the number of people visible in a frame.

    DETERMINISTIC CLASSIFIER PATH ONLY — no generative model, no free-text parsing.
    Character counting must always return an int directly from a detector, never by
    parsing a VLM's prose response. See DECISIONS.md §16.

    Detector chain (in priority order):
      1. OpenCV FaceDetectorYN / YuNet ONNX (OpenCV 5+)
      2. OpenCV DNN SSD ResNet10 (OpenCV 4)
      3. OpenCV Haar cascade frontal+profile (OpenCV 4)
      4. 0 (last resort — no detector available)

    Args:
        image: HxWx3 RGB numpy array, or None (returns 0).

    Returns:
        int: Number of detected people/faces (≥0). Returns 0 for None/black images.
    """
    if image is None or image.max() < 5:
        return 0

    # Deterministic OpenCV detector — no VLM, no free-text parsing.
    return _opencv_count_characters(image)


def _opencv_count_characters(image: np.ndarray) -> int:
    """
    OpenCV-based face count fallback.

    Supports three sub-paths depending on OpenCV version and available models:
      - YuNet (cv2.FaceDetectorYN, OpenCV 5+): best recall, handles poses
      - DNN SSD ResNet10 (cv2.dnn, OpenCV 4): good recall
      - Haar cascade frontal+profile (OpenCV 4): basic recall
    """
    _init_detector()
    try:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ---- YuNet (OpenCV 5+) -------------------------------------------
        if _detector_mode == "yunet" and _yunet is not None:
            h, w = bgr.shape[:2]
            _yunet.setInputSize((w, h))
            _, results = _yunet.detect(bgr)
            if results is None:
                return 0
            # Filter by confidence (last column)
            count = sum(
                1 for det in results if float(det[-1]) >= _YUNET_CONF_THRESHOLD
            )
            return count

        # ---- DNN SSD ResNet10 (OpenCV 4) ----------------------------------
        elif _detector_mode == "dnn" and _face_net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            _face_net.setInput(blob)
            detections = _face_net.forward()
            count = sum(
                1 for i in range(detections.shape[2])
                if float(detections[0, 0, i, 2]) > _DNN_CONF_THRESHOLD
            )
            return count

        # ---- Haar cascade (OpenCV 4) --------------------------------------
        elif _detector_mode in ("opencv_haar_frontal+profile", "opencv_haar_frontal"):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # improves recall in backlit footage

            raw_front = _haar_frontal.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
            )
            faces_front = list(map(tuple, raw_front)) if len(raw_front) > 0 else []

            faces_profile: List[Tuple[int, int, int, int]] = []
            if _haar_profile is not None:
                raw_left = _haar_profile.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
                )
                faces_profile += list(map(tuple, raw_left)) if len(raw_left) > 0 else []

                # Right-facing profiles via horizontal flip + mirror x-coords back
                flipped = cv2.flip(gray, 1)
                w = gray.shape[1]
                raw_right = _haar_profile.detectMultiScale(
                    flipped, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
                )
                if len(raw_right) > 0:
                    for (x, y, bw, bh) in raw_right:
                        faces_profile.append((w - x - bw, y, bw, bh))

            return _dedup_and_count(faces_front, faces_profile)

        else:
            return 0

    except Exception as exc:
        logger.debug("AESE character_stub: detection error: %s", exc)
        return 0
