"""
asvl/asvl/features/novelty.py
Novelty detector — V1 baseline: histogram distance vs. rolling frame buffer mean.

# TODO: swap for embedding-based novelty (e.g. CLIP or CNN features) in a future module.
"""
import cv2
import numpy as np

_HIST_BINS = 64


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV histogram for a single frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    return np.concatenate(hists).astype(np.float32)


def compute_novelty(curr: np.ndarray, buffer_mean_hist: np.ndarray) -> float:
    """
    Compute a novelty score in [0, 1] for the current frame against
    the mean histogram of frames in the rolling buffer.

    Uses chi-squared distance between histograms, capped to [0, 1].

    Args:
        curr: Current frame, HxWx3 uint8 RGB.
        buffer_mean_hist: Mean histogram from RollingFrameBuffer.mean_histogram(),
                          shape (3 * _HIST_BINS,) float32. If None or empty,
                          returns 1.0 (maximally novel — no history yet).

    Returns:
        float in [0, 1] where 0 = identical to buffer average, 1 = maximally novel.
    """
    if buffer_mean_hist is None or buffer_mean_hist.size == 0:
        return 1.0

    curr_hist = _frame_histogram(curr)

    # Chi-squared distance: 0 = identical, larger = more different
    chi_sq = float(cv2.compareHist(
        curr_hist.reshape(-1, 1),
        buffer_mean_hist.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CHISQR,
    ))

    # Empirical max chi-sq for _HIST_BINS * 3 normalized histograms ≈ 3.0
    # Normalize and clip
    score = float(np.clip(chi_sq / 3.0, 0.0, 1.0))
    return score
