"""
asvl/asvl/features/novelty.py
Novelty detector — V1 baseline: histogram distance vs. rolling frame buffer mean.

v1.1 fixes:
  1. Empty-buffer cold-start now returns 0.0 ("not novel") instead of 1.0.
     Returning 1.0 caused all frames to appear maximally novel before the buffer
     populated, saturating the novelty signal and masking real novelty spikes.
  2. Switched from chi-squared distance (unbounded, hard to normalize robustly)
     to correlation-based distance: novelty = 1 - max(0, correl) ∈ [0, 1].
     Correlation is always in [-1, 1]; clipping negatives and inverting gives a
     clean, bounded novelty score without an empirical normalization cap.

# TODO: swap for embedding-based novelty (e.g. CLIP or CNN features) in a future module.
"""
import cv2
import numpy as np

_HIST_BINS = 64
# Thumbnail short-edge for histogram computation — must match buffer.py's _HIST_THUMB_SHORT_EDGE
# so novelty comparison is apples-to-apples (buffer mean hist ↔ current frame hist).
_HIST_THUMB_SHORT_EDGE = 90


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV histogram for a single frame (at thumbnail resolution)."""
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


def compute_novelty(curr: np.ndarray, buffer_mean_hist: np.ndarray) -> float:
    """
    Compute a novelty score in [0, 1] for the current frame against
    the mean histogram of frames in the rolling buffer.

    Uses histogram correlation distance: novelty = 1 - max(0, correl),
    where correl ∈ [-1, 1] from cv2.HISTCMP_CORREL.

    Args:
        curr: Current frame, HxWx3 uint8 RGB.
        buffer_mean_hist: Mean histogram from RollingFrameBuffer.mean_histogram(),
                          shape (3 * _HIST_BINS,) float32. If None or empty,
                          returns 0.0 — the buffer hasn't populated yet so we
                          have no reference to compare against; "not novel" is
                          the safe cold-start default (avoids constant 1.0
                          saturation during the first buffer_seconds of video).

    Returns:
        float in [0, 1] where 0 = identical to buffer average, 1 = maximally novel.
    """
    # v1.1: cold-start safe default is 0.0, NOT 1.0
    if buffer_mean_hist is None or buffer_mean_hist.size == 0:
        return 0.0

    curr_hist = _frame_histogram(curr)

    # Correlation in [-1, 1]: 1.0 = identical histograms, -1.0 = opposite.
    # Clip negative correlation to 0 before inverting so novelty stays in [0, 1].
    correl = float(cv2.compareHist(
        curr_hist.reshape(-1, 1),
        buffer_mean_hist.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))
    novelty = float(1.0 - max(0.0, correl))
    return novelty
