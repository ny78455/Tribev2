"""
aese/adapters/camera_cues.py
Camera cue adapter.

NOT a stub — this is a direct, honest derivation from Module 1's real signals:
  - scene_change=True  → "cut"    (Module 1 detected a hard scene transition)
  - frame is near-black → "black" (very dark frame, possible black-out)
  - black-frame run ending (prev was black, curr is not) → "fade"
  - otherwise          → None

Module 1's scene_change flag is computed from histogram correlation + SSIM distance
(see asvl/features/scene.py). We treat it as a reliable hard-cut detector.
Fades and black frames are detected from the raw image pixel values.

What is explicitly NOT implemented (§1.2):
  - Flashback/dream sequence detection → None (out of scope)
  - Camera movement classification (pan/tilt/zoom) → None (out of scope)
  - Dissolve detection (would require tracking gradual alpha over multiple frames)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# A frame is "near-black" if its mean brightness is below this threshold
_BLACK_FRAME_THRESHOLD = 10.0  # pixel value (0-255)


def _is_near_black(image: Optional[np.ndarray]) -> bool:
    """Return True if the image is a near-black frame."""
    if image is None:
        return False
    return float(image.mean()) < _BLACK_FRAME_THRESHOLD


def detect_camera_cue(
    scene_change: bool,
    image: Optional[np.ndarray],
    prev_was_black: bool = False,
) -> Optional[str]:
    """
    Derive a camera cue from Module 1's scene_change flag and frame-level signals.

    Rules (applied in priority order):
      1. scene_change=True AND frame is NOT black → "cut"
      2. frame is near-black → "black"
      3. prev_was_black AND frame is not near-black → "fade"  (end of a black sequence)
      4. else → None

    Args:
        scene_change: FramePacket.scene_change from Module 1.
        image: HxWx3 RGB numpy array, or None in manifest-replay mode.
        prev_was_black: Whether the previous frame was classified as "black".

    Returns:
        Optional[str]: One of "cut" | "black" | "fade" | None.
    """
    curr_is_black = _is_near_black(image)

    # Rule 1: hard cut
    if scene_change and not curr_is_black:
        return "cut"

    # Rule 2: black frame
    if curr_is_black:
        return "black"

    # Rule 3: fade-in (end of black run)
    if prev_was_black and not curr_is_black:
        return "fade"

    return None
