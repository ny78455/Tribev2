"""
aese/adapters/character_stub.py
Character presence adapter.

V3: Strengthened OpenCV Haar fallback — adds profile-face cascade (checked
on both original and horizontally-flipped frame), looser detection parameters,
and histogram equalization for backlit/high-contrast footage.

V2: Primary path uses riddhimanrana/fastvlm-0.5b-captions (FastVLM) to count
people in a frame. This replaces the OpenCV face-detection stub for frames
where the VLM model is available, giving substantially better recall on
partial faces, side profiles, and low-resolution frames.

Fallback chain:
  1. FastVLM (riddhimanrana/fastvlm-0.5b-captions) — primary
  2. OpenCV DNN SSD ResNet10 face detector — if model files present
  3. OpenCV Haar cascade (frontal + profile) — bundled with OpenCV
  4. 0 — last resort

No character identity, no tracking, no names, no re-identification.
This is an intentional stub per the contract (§1.2, §5.0).
See DECISIONS.md §4.

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
# OpenCV DNN face detector (SSD ResNet10) — loaded once at first call.
# Falls back to Haar cascade (frontal + profile) if DNN model files are not
# available. Falls back to 0 (no face detection) if neither is available.
# ---------------------------------------------------------------------------
_face_net = None
_haar_frontal = None   # haarcascade_frontalface_default.xml
_haar_profile = None   # haarcascade_profileface.xml (left-facing)
_detector_mode: str = "none"  # "dnn" | "opencv_haar_frontal+profile" | "opencv_haar_frontal" | "none"
_detector_init_done = False

# Confidence threshold for DNN detector
_DNN_CONF_THRESHOLD = 0.5

# Paths to look for the DNN face detector model files (relative to this file)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Local bundled cascade files — present in repo under models/haarcascades/.
# Used when the OpenCV install is headless (cv2/data/ contains only __init__.py).
_LOCAL_HAARCASCADES = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "models", "haarcascades")
)
_DNN_PROTO_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "models", "deploy.prototxt"),
    "/usr/share/opencv4/haarcascades/deploy.prototxt",
]
_DNN_MODEL_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "models", "res10_300x300_ssd_iter_140000.caffemodel"),
]


def _init_detector() -> None:
    """Initialize face detector — DNN preferred, Haar cascade (frontal+profile) fallback, 'none' last resort."""
    global _face_net, _haar_frontal, _haar_profile, _detector_mode, _detector_init_done
    if _detector_init_done:
        return
    _detector_init_done = True

    # Try DNN first
    proto = next((p for p in _DNN_PROTO_CANDIDATES if os.path.isfile(p)), None)
    model = next((m for m in _DNN_MODEL_CANDIDATES if os.path.isfile(m)), None)
    if proto and model:
        try:
            _face_net = cv2.dnn.readNet(proto, model)
            _detector_mode = "dnn"
            logger.info("AESE character_stub: DNN face detector loaded.")
            return
        except Exception as exc:
            logger.debug("DNN face detector load failed: %s", exc)

    # Fall back to Haar cascade (bundled with OpenCV) — load frontal + profile.
    # Check cv2.data.haarcascades first; fall back to the local repo copy bundled
    # under models/haarcascades/ (handles opencv-python-headless installs where
    # cv2/data/ only contains __init__.py and no XML files).
    def _find_cascade(filename: str) -> Optional[str]:
        candidates = [
            cv2.data.haarcascades + filename,
            os.path.join(_LOCAL_HAARCASCADES, filename),
        ]
        return next((p for p in candidates if os.path.isfile(p)), None)

    frontal_path = _find_cascade("haarcascade_frontalface_default.xml")
    profile_path = _find_cascade("haarcascade_profileface.xml")

    frontal_ok = frontal_path is not None
    profile_ok = profile_path is not None

    if frontal_ok:
        _haar_frontal = cv2.CascadeClassifier(frontal_path)
        if profile_ok:
            _haar_profile = cv2.CascadeClassifier(profile_path)
            _detector_mode = "opencv_haar_frontal+profile"
            logger.info(
                "AESE character_stub: Haar cascade (frontal + profile) loaded (DNN unavailable)."
            )
        else:
            _detector_mode = "opencv_haar_frontal"
            logger.info(
                "AESE character_stub: Haar cascade (frontal only, no profile XML found) loaded (DNN unavailable)."
            )
        return

    logger.warning(
        "AESE character_stub: No face detector available — count_characters will always return 0. "
        "This is a STUB; see DECISIONS.md §4."
    )
    _detector_mode = "none"


def get_effective_detector_chain() -> str:
    """
    Return a short string describing the active face detection chain.

    Possible values:
      "fastvlm"                   — FastVLM model is loaded and active
      "dnn"                       — OpenCV DNN SSD ResNet10
      "opencv_haar_frontal+profile" — Haar frontal + profile cascades
      "opencv_haar_frontal"       — Haar frontal cascade only
      "none"                      — No detector available

    Triggers detector initialisation if not yet done.
    """
    # Check FastVLM first — it is the primary path
    try:
        from .fastvlm import _fastvlm_available, _ensure_loaded
        _ensure_loaded()
        if _fastvlm_available:
            return "fastvlm"
    except Exception:
        pass

    # Trigger OpenCV initialisation so _detector_mode is populated
    _init_detector()
    return _detector_mode


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
        # Suppress if any already-kept box overlaps significantly
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return len(kept)


def count_characters(image: Optional[np.ndarray]) -> int:
    """
    Count the number of people visible in a frame.

    Primary path: FastVLM (riddhimanrana/fastvlm-0.5b-captions) prompts the VLM
    to count people — handles partial faces, side profiles, and low resolution.
    Fallback: OpenCV DNN / Haar cascade face detectors (frontal + profile).

    Args:
        image: HxWx3 RGB numpy array, or None (returns 0).

    Returns:
        int: Number of detected people/faces (≥0). Returns 0 for None/black images.
    """
    if image is None or image.max() < 5:
        return 0

    # --- Path 1: FastVLM ---
    try:
        from .fastvlm import count_people, _fastvlm_available
        count = count_people(image)
        # If VLM is available and returned a count, trust it
        if _fastvlm_available:
            return count
    except Exception as exc:
        logger.debug("AESE character_stub: FastVLM path failed: %s — trying OpenCV", exc)

    # --- Path 2 & 3: OpenCV face detector (DNN / Haar) ---
    return _opencv_count_characters(image)


def _opencv_count_characters(image: np.ndarray) -> int:
    """
    OpenCV-based face count fallback (DNN SSD or Haar cascade frontal+profile).
    Extracted from count_characters to keep the primary function readable.

    Improvements over V2 Haar path:
      - scaleFactor 1.1 → 1.05 (finer scale steps, better recall)
      - minNeighbors 5 → 4 (less conservative suppression)
      - minSize (30,30) → (20,20) (catches smaller/more distant faces)
      - equalizeHist() applied before detection (improves backlit/high-contrast recall)
      - Profile cascade checked on original AND horizontally-flipped frame
        (haarcascade_profileface.xml is left-facing only)
      - IoU-based deduplication prevents frontal+profile double-counting
    """
    _init_detector()
    try:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if _detector_mode == "dnn":
            blob = cv2.dnn.blobFromImage(
                cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            _face_net.setInput(blob)
            detections = _face_net.forward()
            count = 0
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence > _DNN_CONF_THRESHOLD:
                    count += 1
            return count

        elif _detector_mode in ("opencv_haar_frontal+profile", "opencv_haar_frontal"):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # improves recall in high-contrast/backlit footage

            raw_front = _haar_frontal.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
            )
            faces_front = list(map(tuple, raw_front)) if len(raw_front) > 0 else []

            faces_profile: List[Tuple[int, int, int, int]] = []
            if _haar_profile is not None:
                # Left-facing profiles on the original frame
                raw_left = _haar_profile.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
                )
                faces_profile += list(map(tuple, raw_left)) if len(raw_left) > 0 else []

                # Right-facing profiles: flip the frame, then mirror the x-coordinates back
                flipped = cv2.flip(gray, 1)
                w = gray.shape[1]
                raw_right = _haar_profile.detectMultiScale(
                    flipped, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
                )
                if len(raw_right) > 0:
                    for (x, y, bw, bh) in raw_right:
                        faces_profile.append((w - x - bw, y, bw, bh))  # mirror x back

            return _dedup_and_count(faces_front, faces_profile)

        else:
            return 0

    except Exception as exc:
        logger.debug("AESE character_stub: detection error: %s", exc)
        return 0
