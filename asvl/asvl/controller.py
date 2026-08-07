"""
asvl/asvl/controller.py
Adaptive Sampling Controller.

Implements the weighted importance formula and FPS tier table from §5.6 exactly.
Also computes a human-readable decision_reason from the dominant weighted term.
"""
from typing import Dict


# --- Importance formula (§5.6, verbatim) ---

def compute_importance(
    motion: float,
    scene_change: bool,
    audio_energy: float,
    subtitle_density: float,
    novelty: float,
    weights: Dict[str, float],
) -> float:
    """
    Compute a frame importance score in [0, 1].

    Formula (§5.6):
        score = w_motion * motion
              + w_scene * float(scene_change)
              + w_audio * audio_energy
              + w_subtitle * subtitle_density
              + w_novelty * novelty

    Args:
        motion: Motion score [0, 1].
        scene_change: True if a scene cut was detected.
        audio_energy: Audio RMS energy [0, 1].
        subtitle_density: Fraction of window covered by subtitles [0, 1].
        novelty: Novelty score [0, 1].
        weights: Dict with keys "motion", "scene", "audio", "subtitle", "novelty".

    Returns:
        Clipped importance score in [0, 1].
    """
    score = (
        weights["motion"] * motion
        + weights["scene"] * float(scene_change)
        + weights["audio"] * audio_energy
        + weights["subtitle"] * subtitle_density
        + weights["novelty"] * novelty
    )
    return min(max(score, 0.0), 1.0)


# --- FPS tier table (§5.6, verbatim) ---

def importance_to_fps(score: float) -> float:
    """
    Map an importance score to a target FPS tier.

    Tier table (§5.6):
        score < 0.2 → 0.5 fps
        score < 0.4 → 1.0 fps
        score < 0.6 → 2.0 fps
        score < 0.8 → 5.0 fps
        score ≥ 0.8 → 10.0 fps

    Args:
        score: Importance score in [0, 1].

    Returns:
        Target FPS (float).
    """
    if score < 0.2:
        return 0.5
    if score < 0.4:
        return 1.0
    if score < 0.6:
        return 2.0
    if score < 0.8:
        return 5.0
    return 10.0


# --- Decision reason (dominant weighted term) ---

_REASON_MAP = {
    "motion": "Fast motion",
    "scene": "Scene transition",
    "audio": "High audio energy",
    "subtitle": "Active subtitles",
    "novelty": "Novel content",
}

_STATIC_REASON = "Static / dialogue"


def get_decision_reason(
    motion: float,
    scene_change: bool,
    audio_energy: float,
    subtitle_density: float,
    novelty: float,
    weights: Dict[str, float],
    importance_score: float,
) -> str:
    """
    Return a human-readable label for the dominant reason a frame was kept.

    Computes the weighted contribution of each signal and picks the largest.
    Returns "Static / dialogue" for very low-importance frames.

    Args:
        All signal values and weights as per compute_importance().
        importance_score: The already-computed importance score.

    Returns:
        A short descriptive string.
    """
    if importance_score < 0.2:
        return _STATIC_REASON

    contributions = {
        "motion": weights["motion"] * motion,
        "scene": weights["scene"] * float(scene_change),
        "audio": weights["audio"] * audio_energy,
        "subtitle": weights["subtitle"] * subtitle_density,
        "novelty": weights["novelty"] * novelty,
    }
    dominant = max(contributions, key=contributions.__getitem__)
    return _REASON_MAP.get(dominant, _STATIC_REASON)
