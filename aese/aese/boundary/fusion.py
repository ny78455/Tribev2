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

Manifest-replay mode (no --video):
  When image-dependent channels (scene, character, embedding, prediction_error)
  have no real data, their weights are excluded from BOTH numerator and denominator.
  This renormalizes the fused score over only the available channels so that
  camera + dialogue + music can still cross boundary_threshold even without image data.
  Without this fix, those channels would silently contribute 0 to the numerator while
  still consuming their full weight in the denominator — structurally capping the
  fused score at ~0.39 (camera+dialogue+music weights) and making any threshold≥0.40
  impossible to cross.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..types import BoundarySignal

# Maps config weight keys → the effective contribution of that key on BoundarySignal.
# "character" and "camera" share one weight slot (fused as max); handled explicitly below.
_WEIGHT_KEYS = frozenset(
    {"prediction_error", "scene", "dialogue", "emotion", "character", "embedding", "music"}
)


def fuse(
    signals: BoundarySignal,
    weights: dict,
    available: Optional[Dict[str, bool]] = None,
) -> float:
    """
    V1 weighted-sum fusion of boundary signals, with weight renormalization
    over available channels.

    # TODO: swap for learned MLP/small transformer per Section 15.

    Args:
        signals: BoundarySignal dataclass with all signal values in [0, 1].
        weights: dict from AESEConfig.weights — must sum to ~1.0.
        available: Optional dict mapping weight-key → bool (True = real data backed
                   this channel). Missing keys default to True (assumed available).
                   When a channel is False, its weight is excluded from both numerator
                   and denominator so the score is not structurally capped.

    Returns:
        float: Fused score in [0, 1]. Values > boundary_threshold trigger boundary detection.
    """
    if available is None:
        available = {}

    def _is_available(key: str) -> bool:
        return available.get(key, True)

    # character and camera share the "character" weight slot.
    # Fused as max(character, camera) to avoid double-counting.
    # The slot is "available" if either signal has real data:
    #   - camera is always available (derived from scene_change, not pixel data)
    #   - character is available only when image data exists
    # So the combined slot is always available (camera alone suffices).
    char_cam_contribution = max(signals.character, signals.camera)

    # Build per-key contributions, skipping unavailable channels.
    active_total_weight = 0.0
    score = 0.0

    for key in ("prediction_error", "scene", "dialogue", "emotion", "embedding", "music"):
        if not _is_available(key):
            continue
        w = weights.get(key, 0.0)
        active_total_weight += w
        if key == "prediction_error":
            score += w * signals.prediction_error
        elif key == "scene":
            score += w * signals.scene
        elif key == "dialogue":
            score += w * signals.dialogue
        elif key == "emotion":
            score += w * signals.emotion  # always 0.0 * w — see DECISIONS.md §8
        elif key == "embedding":
            score += w * signals.embedding_distance
        elif key == "music":
            score += w * signals.music

    # Always include the character/camera shared slot (camera is always available)
    char_cam_weight = weights.get("character", 0.0)
    active_total_weight += char_cam_weight
    score += char_cam_weight * char_cam_contribution

    if active_total_weight == 0.0:
        return 0.0

    # Renormalize so available channels use their full weight budget
    normalized = score / active_total_weight

    # Clamp to [0, 1] for numerical safety
    return float(min(max(normalized, 0.0), 1.0))


def dominant_signal_name(
    signals: BoundarySignal,
    weights: dict,
    available: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Identify which signal contributed the most to the fused score.
    Used to populate Event.boundary_reason for human-readable explanations.

    Args:
        signals: BoundarySignal dataclass.
        weights: dict from AESEConfig.weights.
        available: Optional availability dict (same semantics as fuse()).

    Returns:
        str: Name of the dominant signal (e.g. "scene", "embedding", "dialogue").
    """
    if available is None:
        available = {}

    def _is_available(key: str) -> bool:
        return available.get(key, True)

    contributions = {}
    if _is_available("prediction_error"):
        contributions["prediction_error"] = weights.get("prediction_error", 0.0) * signals.prediction_error
    if _is_available("scene"):
        contributions["scene"] = weights.get("scene", 0.0) * signals.scene
    if _is_available("dialogue"):
        contributions["dialogue"] = weights.get("dialogue", 0.0) * signals.dialogue
    if _is_available("emotion"):
        contributions["emotion"] = weights.get("emotion", 0.0) * signals.emotion
    # camera and character share the weight slot; camera dominates when it fires
    # camera is always available
    contributions["camera"] = weights.get("character", 0.0) * signals.camera
    if _is_available("character"):
        contributions["character"] = weights.get("character", 0.0) * signals.character
    if _is_available("embedding"):
        contributions["embedding"] = weights.get("embedding", 0.0) * signals.embedding_distance
    if _is_available("music"):
        contributions["music"] = weights.get("music", 0.0) * signals.music

    if not contributions:
        return "none"
    return max(contributions, key=contributions.get)
