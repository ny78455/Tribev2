"""
aese/boundary/fusion.py
Multi-cue boundary fusion — combines all boundary signals into a single score.

V1: Weighted sum of all signals using renormalized weights from AESEConfig.
This is the correct V1 per §5.6:
  "weighted sum (matches the config weights, renormalized per the Section 4 fix)"

# TODO: swap for learned MLP/small transformer per Section 15.
  The fuse() interface stays fixed; the internal fusion can be replaced by any
  differentiable function of BoundarySignal → float.

Note: The emotion signal always contributes 0.0 to this sum (no emotion model).
This means the effective weight of the emotion dimension is always zero in practice,
and the other signals bear proportionally more influence. This is by design — see
DECISIONS.md §8.
"""
from __future__ import annotations

from ..types import BoundarySignal


def fuse(signals: BoundarySignal, weights: dict) -> float:
    """
    V1 weighted-sum fusion of boundary signals.

    # TODO: swap for learned MLP/small transformer per Section 15.

    Args:
        signals: BoundarySignal dataclass with all signal values in [0, 1].
        weights: dict from AESEConfig.weights — must sum to ~1.0.

    Returns:
        float: Fused score in [0, 1]. Values > boundary_threshold trigger boundary detection.
    """
    score = (
        weights.get("prediction_error", 0.0) * signals.prediction_error
        + weights.get("scene", 0.0) * signals.scene
        + weights.get("dialogue", 0.0) * signals.dialogue
        + weights.get("emotion", 0.0) * signals.emotion   # always 0.0 * signals.emotion = 0.0
        # character and camera share the "character" weight dimension:
        # camera cue is a harder cut signal; character is a softer change signal.
        # Fused as max(character, camera) to avoid double-counting.
        + weights.get("character", 0.0) * max(signals.character, signals.camera)
        + weights.get("embedding", 0.0) * signals.embedding_distance
        + weights.get("music", 0.0) * signals.music
    )
    # Clamp to [0, 1] for numerical safety
    return float(min(max(score, 0.0), 1.0))


def dominant_signal_name(signals: BoundarySignal, weights: dict) -> str:
    """
    Identify which signal contributed the most to the fused score.
    Used to populate Event.boundary_reason for human-readable explanations.

    Returns:
        str: Name of the dominant signal (e.g. "scene", "embedding", "dialogue").
    """
    contributions = {
        "prediction_error": weights.get("prediction_error", 0.0) * signals.prediction_error,
        "scene": weights.get("scene", 0.0) * signals.scene,
        "dialogue": weights.get("dialogue", 0.0) * signals.dialogue,
        "emotion": weights.get("emotion", 0.0) * signals.emotion,
        # camera and character share the weight slot; camera dominates when it fires
        "camera": weights.get("character", 0.0) * signals.camera,
        "character": weights.get("character", 0.0) * signals.character,
        "embedding": weights.get("embedding", 0.0) * signals.embedding_distance,
        "music": weights.get("music", 0.0) * signals.music,
    }
    return max(contributions, key=contributions.get)
