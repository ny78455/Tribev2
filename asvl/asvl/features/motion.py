"""
asvl/asvl/features/motion.py
Optical flow motion score using Farneback dense optical flow.

v1.1 fixes:
1. Replaced whole-frame mean magnitude with 90th-percentile (p90) statistic.
   A moving subject against a static background no longer gets diluted.
   Old formula: score = min(mean_magnitude / 50.0, 1.0)
   New formula: score = min(p90 / MOTION_NORM_CONST, 1.0)
   See DECISIONS.md §12.

2. Performance: optical flow is computed on a downsampled thumbnail
   (_FLOW_SHORT_EDGE = 90 px) rather than on full-resolution frames.
   On 720×1280 input this reduces flow computation from ~134ms to ~3ms per
   frame (44×), enabling the 60s per-clip runtime target. The p90 statistic
   is invariant to resolution scaling (subject-to-background ratio is
   preserved). See DECISIONS.md §16.
"""
import cv2
import numpy as np

# Normalization constant for the 90th-percentile flow magnitude at thumbnail resolution.
# Tuned against comedy.mp4 clip p90 distribution:
#   - Dialogue (0-30s): median p90 = 1.59, p90-of-p90 = 6.09
#   - Action (31-60s): median p90 = 3.65, p90-of-p90 = 7.80
#
# At MOTION_NORM_CONST = 10.0:
#   - Dialogue median score = 1.59/10 = 0.16 → 0.5/1.0 fps tier ✓
#   - Action p90 score = 7.80/10 = 0.78 → 5.0 fps tier ✓
#
# Old values: _EMPIRICAL_MAX_MAGNITUDE = 50.0 (full-res mean, v1.0)
#             MOTION_NORM_CONST = 2.0 (thumbnail p90, too low — caused dialogue saturation)
MOTION_NORM_CONST = 10.0

# Short edge of the thumbnail used for optical flow computation.
_FLOW_SHORT_EDGE = 90


def _thumbnail_gray(frame: np.ndarray) -> np.ndarray:
    """
    Downsample a frame to _FLOW_SHORT_EDGE on the short axis, convert to gray.

    Returns the frame as-is (gray) if it's already smaller than the target.
    """
    h, w = frame.shape[:2]
    if h <= _FLOW_SHORT_EDGE and w <= _FLOW_SHORT_EDGE:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    scale = _FLOW_SHORT_EDGE / min(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def compute_motion_score(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Compute a motion score in [0, 1] between two consecutive RGB frames.

    Uses Farneback dense optical flow on downsampled grayscale thumbnails.
    The 90th-percentile pixel magnitude is normalized by MOTION_NORM_CONST,
    so that a fast subject occupying ~10% of the frame area produces a high
    score even when the rest of the background is static.

    Args:
        prev: Previous frame, HxWx3 uint8 RGB.
        curr: Current frame, HxWx3 uint8 RGB.

    Returns:
        float in [0, 1] where 0 = no motion, 1 = maximum expected motion.
    """
    prev_gray = _thumbnail_gray(prev)
    curr_gray = _thumbnail_gray(curr)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # 90th-percentile of moving pixels — not the whole-frame mean —
    # so a fast subject against a static background is not diluted.
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    p90 = float(np.percentile(magnitude, 90))
    return float(min(p90 / MOTION_NORM_CONST, 1.0))
