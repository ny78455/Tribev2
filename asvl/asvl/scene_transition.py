"""
asvl/asvl/scene_transition.py
Scene Transition Detector.

Wraps histogram + SSIM scene scoring (from features/scene.py) and black-frame
detection (from features/blur.py) to emit a boolean scene_changed per frame.

This is intentionally independent of the continuous scene_score used in the
importance formula — it feeds the scene_change boolean signal separately.
"""
import logging
from typing import Optional

import numpy as np

from .features.scene import compute_scene_score
from .features.blur import is_black_frame

logger = logging.getLogger(__name__)


class SceneTransitionDetector:
    """
    Stateful detector that tracks the previous frame and detects cuts.

    Decision logic:
        scene_changed = True  if:
            - compute_scene_score() returns scene_change=True, OR
            - current frame is a black frame (inter-scene fade marker)
    """

    def __init__(self, scene_threshold: float = 0.45) -> None:
        """
        Args:
            scene_threshold: Passed to compute_scene_score().
        """
        self._scene_threshold = scene_threshold
        self._prev_frame: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> bool:
        """
        Process a new frame and return whether a scene transition occurred.

        Args:
            frame: Current frame, HxWx3 uint8 RGB.

        Returns:
            True if a scene transition (cut or black frame) was detected.
        """
        if self._prev_frame is None:
            self._prev_frame = frame
            return False

        # Black frame → always a transition marker
        if is_black_frame(frame):
            logger.debug("Scene transition: black frame detected.")
            self._prev_frame = frame
            return True

        # Histogram + SSIM check
        try:
            _, scene_changed = compute_scene_score(
                self._prev_frame, frame, self._scene_threshold
            )
        except Exception as exc:
            logger.warning("Scene score computation failed: %s", exc)
            scene_changed = False

        self._prev_frame = frame
        return bool(scene_changed)

    def reset(self) -> None:
        """Clear internal state (e.g. when seeking to a new position)."""
        self._prev_frame = None
