"""
asvl/asvl/features/scene.py
Scene change detection via histogram correlation distance + SSIM.

v1.1 fixes:
1. hist_dist formula: was (1 - correlation) / 2.0 which halved the signal.
   Corrected to 1 - max(0, correlation) ∈ [0, 1]. See DECISIONS.md §13.

2. Scene threshold lowered from 0.7 to 0.45. See DECISIONS.md §13.

3. Performance: SSIM is now computed on a downsampled thumbnail
   (_SSIM_SHORT_EDGE = 90 px) rather than on full-resolution frames.
   On 720×1280 input this reduces SSIM pixels from ~921 K to ~14 K (64×
   speedup), bringing per-frame latency from ~400ms to ~7ms. See DECISIONS.md §16.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

_HIST_BINS = 64
_DEFAULT_SCENE_THRESHOLD = 0.45  # was 0.7; see DECISIONS.md §13

# SSIM thumbnail short-edge resolution. A 90px thumbnail retains sufficient
# structural information for scene-change detection (hard cuts are global
# appearance changes) while reducing SSIM pixel-count by ~64× on 720p input.
_SSIM_SHORT_EDGE = 90


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized 3-channel (HSV) flattened histogram."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [_HIST_BINS], [0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    return np.concatenate(hists)


def _thumbnail_gray(frame: np.ndarray) -> np.ndarray:
    """
    Downsample a frame to a fixed short-edge resolution and convert to grayscale.

    Aspect ratio is preserved. Short edge is clamped to _SSIM_SHORT_EDGE px.
    Frames already smaller than the target are returned as-is (no upscaling).
    """
    h, w = frame.shape[:2]
    if h <= _SSIM_SHORT_EDGE and w <= _SSIM_SHORT_EDGE:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray
    if h <= w:
        # Height is short edge
        scale = _SSIM_SHORT_EDGE / h
    else:
        scale = _SSIM_SHORT_EDGE / w
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def compute_scene_score(
    prev: np.ndarray,
    curr: np.ndarray,
    scene_threshold: float = _DEFAULT_SCENE_THRESHOLD,
) -> tuple:
    """
    Compute a scene change score in [0, 1] between two consecutive RGB frames.

    Combines:
      - Histogram correlation distance: 1 - max(0, correl(prev_hist, curr_hist))
      - 1 - SSIM (structural similarity, computed on downsampled thumbnails)

    Weighted average: 0.5 * hist_dist + 0.5 * ssim_dist

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
    # Correlation ∈ [-1, 1]. Clip negative values (anti-correlated frames are
    # at least as different as uncorrelated ones), then invert: result ∈ [0, 1].
    # v1.1: was (1.0 - correlation) / 2.0 which incorrectly halved the signal.
    hist_dist = float(np.clip(1.0 - max(0.0, correlation), 0.0, 1.0))

    # SSIM — operate on downsampled grayscale thumbnails for speed.
    # Hard cuts are global appearance changes; thumbnail resolution is sufficient.
    prev_gray = _thumbnail_gray(prev)
    curr_gray = _thumbnail_gray(curr)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    ssim_val = float(ssim(prev_gray, curr_gray, data_range=255))
    ssim_dist = float(np.clip(1.0 - ssim_val, 0.0, 1.0))

    score = float(np.clip(0.5 * hist_dist + 0.5 * ssim_dist, 0.0, 1.0))
    scene_change = score > scene_threshold
    return score, scene_change
