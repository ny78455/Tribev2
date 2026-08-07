"""
asvl/asvl/features/motion.py
Optical flow motion score using Farneback dense optical flow.
"""
import cv2
import numpy as np

# Empirical maximum magnitude used for normalization.
# Farneback flow on typical video has magnitudes well below 50 px/frame.
_EMPIRICAL_MAX_MAGNITUDE = 50.0


def compute_motion_score(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Compute a motion score in [0, 1] between two consecutive RGB frames.

    Uses Farneback dense optical flow on grayscale images.
    The mean magnitude of the flow field is normalized by an empirical maximum.

    Args:
        prev: Previous frame, HxWx3 uint8 RGB.
        curr: Current frame, HxWx3 uint8 RGB.

    Returns:
        float in [0, 1] where 0 = no motion, 1 = maximum expected motion.
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

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

    # Compute per-pixel magnitude
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_magnitude = float(np.mean(magnitude))
    score = min(mean_magnitude / _EMPIRICAL_MAX_MAGNITUDE, 1.0)
    return score
