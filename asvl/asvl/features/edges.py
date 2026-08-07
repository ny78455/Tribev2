"""
asvl/asvl/features/edges.py
Edge difference score using Canny edge maps.
"""
import cv2
import numpy as np


def compute_edge_diff(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    Compute edge difference score in [0, 1] between two consecutive RGB frames.

    Applies Canny edge detection to both frames (grayscale), computes the
    XOR of the binary edge maps, and normalizes by total pixel count.

    Args:
        prev: Previous frame, HxWx3 uint8 RGB.
        curr: Current frame, HxWx3 uint8 RGB.

    Returns:
        float in [0, 1] where 0 = identical edges, 1 = completely different edges.
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

    prev_edges = cv2.Canny(prev_gray, threshold1=50, threshold2=150)
    curr_edges = cv2.Canny(curr_gray, threshold1=50, threshold2=150)

    # XOR: pixels that changed edge status
    xor_map = cv2.bitwise_xor(prev_edges, curr_edges)

    # Normalize by total pixel count
    total_pixels = prev_gray.size
    diff = float(np.count_nonzero(xor_map)) / total_pixels
    return float(np.clip(diff, 0.0, 1.0))
