"""
aese/adapters/character_stub.py
Character presence adapter.

# STUB: OpenCV DNN face detector — returns face count only.
No character identity, no tracking, no names, no re-identification.
This is an intentional stub per the contract (§1.2, §5.0):
  "No character re-identification/face clustering pipeline — stub: return empty list; log as future work."

Face count is used as a coarse proxy for "character entrance/exit" in boundary signals.
Count values are unreliable on small/occluded faces and at low resolution.
See DECISIONS.md §4.

Future work: replace with a proper face-tracking + re-ID pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenCV DNN face detector (SSD ResNet10) — loaded once at first call.
# Falls back to Haar cascade if DNN model files are not available.
# Falls back to 0 (no face detection) if neither is available.
# ---------------------------------------------------------------------------
_face_net = None
_haar_cascade = None
_detector_mode: str = "none"  # "dnn" | "haar" | "none"
_detector_init_done = False

# Confidence threshold for DNN detector
_DNN_CONF_THRESHOLD = 0.5

# Paths to look for the DNN face detector model files (relative to this file)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DNN_PROTO_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "models", "deploy.prototxt"),
    "/usr/share/opencv4/haarcascades/deploy.prototxt",
]
_DNN_MODEL_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "models", "res10_300x300_ssd_iter_140000.caffemodel"),
]


def _init_detector() -> None:
    """Initialize face detector — DNN preferred, Haar cascade fallback, 'none' last resort."""
    global _face_net, _haar_cascade, _detector_mode, _detector_init_done
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

    # Fall back to Haar cascade (bundled with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.isfile(cascade_path):
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
        _detector_mode = "haar"
        logger.info("AESE character_stub: Haar cascade face detector loaded (DNN unavailable).")
        return

    logger.warning(
        "AESE character_stub: No face detector available — count_characters will always return 0. "
        "This is a STUB; see DECISIONS.md §4."
    )
    _detector_mode = "none"


def count_characters(image: Optional[np.ndarray]) -> int:
    """
    # STUB: Count faces in a frame using OpenCV face detector.
    Returns the number of detected faces (≥0).
    No identity, no tracking — just a raw count.

    Args:
        image: HxWx3 RGB numpy array, or None (returns 0).

    Returns:
        int: Number of detected faces. Returns 0 for black frames / None images.
    """
    if image is None or image.max() < 5:
        return 0

    _init_detector()

    try:
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if _detector_mode == "dnn":
            h, w = bgr.shape[:2]
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

        elif _detector_mode == "haar":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = _haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return len(faces) if len(faces) > 0 else 0

        else:
            return 0

    except Exception as exc:
        logger.debug("AESE character_stub: detection error: %s", exc)
        return 0
