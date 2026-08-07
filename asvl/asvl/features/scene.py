"""
asvl/asvl/features/scene.py
Scene change detection via histogram correlation distance + SSIM.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

_HIST_BINS = 64
_DEFAULT_SCENE_THRESHOLD = 0.7


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized 3-channel (HSV) flattened histogram."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    return np.concatenate(hists)


def compute_scene_score(
    prev: np.ndarray,
    curr: np.ndarray,
    scene_threshold: float = _DEFAULT_SCENE_THRESHOLD,
) -> tuple:
    """
    Compute a scene change score in [0, 1] between two consecutive RGB frames.

    Combines:
      - Histogram correlation distance: 1 - correlation(prev_hist, curr_hist)
      - 1 - SSIM (structural similarity)

    Weighted average: 0.5 * hist_dist + 0.5 * (1 - ssim_val)

    Args:
        prev: Previous frame, HxWx3 uint8 RGB.
        curr: Current frame, HxWx3 uint8 RGB.
        scene_threshold: If score > this, scene_change=True.

    Returns:
        (score: float in [0,1], scene_change: bool)
    """
    # Histogram correlation distance
    prev_hist = _compute_histogram(prev)
    curr_hist = _compute_histogram(curr)
    correlation = float(cv2.compareHist(
        prev_hist.reshape(-1, 1).astype(np.float32),
        curr_hist.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))
    # Correlation ranges [-1, 1]; distance = 1 - correlation, clipped to [0,1]
    hist_dist = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))

    # SSIM — work on grayscale, resize if frames differ in size
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    ssim_val = float(ssim(prev_gray, curr_gray, data_range=255))
    ssim_dist = float(np.clip(1.0 - ssim_val, 0.0, 1.0))

    score = float(np.clip(0.5 * hist_dist + 0.5 * ssim_dist, 0.0, 1.0))
    scene_change = score > scene_threshold
    return score, scene_change
