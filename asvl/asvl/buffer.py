"""
asvl/asvl/buffer.py
Rolling Frame Buffer backed by collections.deque.
Stores (frame, timestamp_ms, FrameFeatures) tuples.
Window size = buffer_seconds * fps.

v1.1 performance fix: mean_histogram() now uses an incrementally maintained
histogram sum rather than recomputing all frame histograms on every call.
This reduces mean_histogram() from O(buffer_size) to O(1) per call.
See DECISIONS.md §17.
"""
import collections
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import FrameFeatures

logger = logging.getLogger(__name__)

_HIST_BINS = 64
# Thumbnail size for histogram computation — matches scene.py / novelty.py for consistency
_HIST_THUMB_SHORT_EDGE = 90


def _frame_histogram_thumb(frame: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 3-channel HSV histogram on a downsampled thumbnail.

    Downsampling to _HIST_THUMB_SHORT_EDGE px short-edge before histogram
    computation reduces cost from ~1.9ms to ~0.4ms at 720×1280 while
    preserving color distribution for novelty comparison.
    """
    h, w = frame.shape[:2]
    if h > _HIST_THUMB_SHORT_EDGE or w > _HIST_THUMB_SHORT_EDGE:
        scale = _HIST_THUMB_SHORT_EDGE / min(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hists = []
    for ch in range(3):
        h_arr = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
        cv2.normalize(h_arr, h_arr)
        hists.append(h_arr.flatten())
    return np.concatenate(hists).astype(np.float32)


class RollingFrameBuffer:
    """
    A fixed-size rolling buffer of decoded frames with their features.

    Attributes:
        maxlen: Maximum number of frames stored.
        _deque: Internal deque of (frame, timestamp_ms, FrameFeatures) tuples.
        _hist_deque: Parallel deque of pre-computed per-frame histograms.
        _hist_sum: Running sum of histograms for O(1) mean_histogram().
    """

    def __init__(self, buffer_seconds: float, fps: float) -> None:
        self.maxlen = max(1, int(buffer_seconds * fps))
        self._deque: collections.deque = collections.deque(maxlen=self.maxlen)
        # Parallel histogram deque — same maxlen, for incremental mean tracking
        self._hist_deque: collections.deque = collections.deque(maxlen=self.maxlen)
        self._hist_sum: Optional[np.ndarray] = None  # sum of all histograms in buffer

    def push(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        features: FrameFeatures,
    ) -> None:
        """Add a frame and its features to the buffer, maintaining histogram sum."""
        # Compute histogram for this frame
        try:
            new_hist = _frame_histogram_thumb(frame)
        except Exception as exc:
            logger.debug("Histogram computation error in push: %s", exc)
            new_hist = np.zeros(3 * _HIST_BINS, dtype=np.float32)

        # If buffer is at capacity, subtract the histogram being evicted
        if len(self._hist_deque) == self.maxlen:
            evicted = self._hist_deque[0]
            if self._hist_sum is not None:
                self._hist_sum -= evicted

        # Add new histogram to sum
        if self._hist_sum is None:
            self._hist_sum = new_hist.copy()
        else:
            self._hist_sum = self._hist_sum + new_hist

        self._hist_deque.append(new_hist)
        self._deque.append((frame, timestamp_ms, features))

    def __len__(self) -> int:
        return len(self._deque)

    def mean_histogram(self) -> Optional[np.ndarray]:
        """
        Return the mean HSV histogram over all frames currently in the buffer.

        O(1) — uses the incrementally maintained histogram sum.

        Returns:
            Mean histogram as float32 array of shape (3 * _HIST_BINS,),
            or None if the buffer is empty.
        """
        n = len(self._deque)
        if n == 0 or self._hist_sum is None:
            return None
        return (self._hist_sum / n).astype(np.float32)

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
