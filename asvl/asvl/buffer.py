"""
asvl/asvl/buffer.py
Rolling Frame Buffer backed by collections.deque.
Stores (frame, timestamp_ms, FrameFeatures) tuples.
Window size = buffer_seconds * fps.
"""
import collections
import logging
from typing import List, Optional, Tuple

import numpy as np

from .types import FrameFeatures

logger = logging.getLogger(__name__)

_HIST_BINS = 64


class RollingFrameBuffer:
    """
    A fixed-size rolling buffer of decoded frames with their features.

    Attributes:
        maxlen: Maximum number of frames stored.
        _deque: Internal deque of (frame, timestamp_ms, FrameFeatures) tuples.
    """

    def __init__(self, buffer_seconds: float, fps: float) -> None:
        self.maxlen = max(1, int(buffer_seconds * fps))
        self._deque: collections.deque = collections.deque(maxlen=self.maxlen)

    def push(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        features: FrameFeatures,
    ) -> None:
        """Add a frame and its features to the buffer."""
        self._deque.append((frame, timestamp_ms, features))

    def __len__(self) -> int:
        return len(self._deque)

    def mean_histogram(self) -> Optional[np.ndarray]:
        """
        Compute the mean HSV histogram over all frames currently in the buffer.

        Returns:
            Mean histogram as float32 array of shape (3 * _HIST_BINS,),
            or None if the buffer is empty.
        """
        if not self._deque:
            return None

        hists = []
        for frame, _, _ in self._deque:
            try:
                import cv2
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                frame_hist = []
                for ch in range(3):
                    h = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
                    cv2.normalize(h, h)
                    frame_hist.append(h.flatten())
                hists.append(np.concatenate(frame_hist))
            except Exception as exc:
                logger.debug("Histogram computation error: %s", exc)
                continue

        if not hists:
            return None

        return np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32)

    def recent_importance_scores(self) -> List[float]:
        """
        Return a list of importance scores for frames currently in the buffer.
        Computed as motion_score for simplicity (proxy for overall importance).

        Returns:
            List of floats, one per buffered frame.
        """
        return [feat.motion_score for _, _, feat in self._deque]

    def frames(self) -> List[Tuple[np.ndarray, float, FrameFeatures]]:
        """Return all buffered (frame, timestamp_ms, features) tuples as a list."""
        return list(self._deque)
