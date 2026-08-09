"""
aese/adapters/action_stub.py
Action label adapter.

# STUB: Bucket motion_score into a 3-tier action label.
NOT a real action recognition model (no optical flow classification, no pose estimation,
no temporal convolutions). This is an intentional stub per §5.0:
  "threshold bucket (motion_score<0.2→'static', <0.5→'walking', else→'fast_action')"

Used as a proxy to drive the event_classifier rule-based logic downstream.
A real action recognition model (e.g. SlowFast, X3D) can replace this behind
the same interface in a future module.
See DECISIONS.md §5.
"""
from __future__ import annotations

_ACTION_THRESHOLDS = [
    (0.2, "static"),
    (0.5, "walking"),
    (1.1, "fast_action"),  # catch-all for scores ≥ 0.5
]


def label_action(motion_score: float) -> str:
    """
    # STUB: Bucket motion_score into a coarse action label.

    Thresholds:
        motion_score < 0.2  → "static"
        motion_score < 0.5  → "walking"
        else                → "fast_action"

    Args:
        motion_score: float in [0, 1], from FramePacket.motion_score (Module 1).

    Returns:
        str: One of "static" | "walking" | "fast_action".
    """
    for threshold, label in _ACTION_THRESHOLDS:
        if motion_score < threshold:
            return label
    return "fast_action"  # should never reach here
