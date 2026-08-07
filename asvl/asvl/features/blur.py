"""
asvl/asvl/features/blur.py
Blur / fade / black-frame detection.
"""
import cv2
import numpy as np

# Laplacian variance below this → frame considered blurred
_BLUR_LAPLACIAN_THRESHOLD = 50.0

# Mean pixel intensity below this → frame considered black
_BLACK_FRAME_INTENSITY_THRESHOLD = 10.0


def is_blurred(frame: np.ndarray) -> bool:
    """
    Detect whether a frame is blurred or in a fade transition.

    Uses the variance of the Laplacian of the grayscale image.
    Low variance (< _BLUR_LAPLACIAN_THRESHOLD) indicates blurring.

    Args:
        frame: HxWx3 uint8 RGB image.

    Returns:
        True if frame is blurred, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return laplacian_var < _BLUR_LAPLACIAN_THRESHOLD


def is_black_frame(frame: np.ndarray) -> bool:
    """
    Detect whether a frame is a black (or near-black) frame.

    Uses the mean pixel intensity across all channels.
    Mean below _BLACK_FRAME_INTENSITY_THRESHOLD → black frame.

    Args:
        frame: HxWx3 uint8 RGB image.

    Returns:
        True if frame is a black frame, False otherwise.
    """
    mean_intensity = float(np.mean(frame))
    return mean_intensity < _BLACK_FRAME_INTENSITY_THRESHOLD
