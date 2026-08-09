"""
aese/boundary/signals.py
Named boundary signal functions — each returns a float in [0, 1].

Per §5.3, each signal captures one dimension of evidence that a meaningful
event boundary has occurred between the previous and current second.

CRITICAL HONESTY CHECKPOINT (§5.3):
  emotion_signal always returns 0.0. There is NO emotion model in scope for
  Module 2 and Module 1 provides NO emotion-related input signal. Returning a
  fabricated non-zero value would silently corrupt every downstream boundary
  decision. Return 0.0 and document it — do not invent a proxy.
  See DECISIONS.md §8.
"""
from __future__ import annotations

from typing import Optional

from ..types import TemporalFeature


# ---------------------------------------------------------------------------
# A: Scene signal
# ---------------------------------------------------------------------------
def scene_signal(curr: TemporalFeature, prev: TemporalFeature) -> float:
    """
    Returns 1.0 if the scene label changed between consecutive seconds.
    NOTE: This relies on the STUB scene_label adapter — see adapters/scene_label.py.
    Precision is limited (~60-70%); see DECISIONS.md §3.
    """
    return 1.0 if curr.scene_label != prev.scene_label else 0.0


# ---------------------------------------------------------------------------
# B: Character signal
# ---------------------------------------------------------------------------
def character_signal(curr: TemporalFeature, prev: TemporalFeature) -> float:
    """
    Returns a normalized character-count change in [0, 1].
    Uses abs delta / 2.0, capped at 1.0.
    STUB proxy for "character entrance/exit" — face count is unreliable;
    see adapters/character_stub.py.

    Normalization constant 2.0: a delta of 2 faces is considered a maximal change
    for most movie scenes. Larger deltas saturate at 1.0.
    """
    return min(abs(curr.character_count - prev.character_count) / 2.0, 1.0)


# ---------------------------------------------------------------------------
# C: Dialogue signal
# ---------------------------------------------------------------------------
def dialogue_signal(curr: TemporalFeature, prev: TemporalFeature) -> float:
    """
    Returns 1.0 if dialogue presence flipped (onset or cessation of speech).
    Strong event boundary indicator: conversations start/end typically mark scene shifts.
    """
    return 1.0 if curr.dialogue_present != prev.dialogue_present else 0.0


# ---------------------------------------------------------------------------
# D: Camera signal
# ---------------------------------------------------------------------------
def camera_signal(curr: TemporalFeature) -> float:
    """
    Returns 1.0 if a camera cut or fade is detected in the current second.
    Derived from Module 1's scene_change flag via camera_cues adapter — not a stub.
    "black" frames alone do not trigger this (they precede cuts, not mark them).
    """
    return 1.0 if curr.camera_cue in ("cut", "fade") else 0.0


# ---------------------------------------------------------------------------
# E: Emotion signal — INTENTIONAL STUB, ALWAYS RETURNS 0.0
# ---------------------------------------------------------------------------
def emotion_signal(curr: TemporalFeature, prev: TemporalFeature) -> float:
    """
    # STUB: No emotion model is in scope for Module 2.
    Module 1 provides no emotion-related signal.
    ALWAYS returns 0.0.

    DO NOT "helpfully" invent an emotion proxy here — any fabricated value
    (e.g. using audio_energy as an emotion proxy) would corrupt boundary decisions
    with a signal that has no semantic relationship to actual emotion transitions.
    This is a documented zero, not a missing feature.

    See DECISIONS.md §8.
    Future work: replace with a lightweight valence/arousal model (e.g. fine-tuned
    CLIP on FER-2013 or AffectNet).
    """
    return 0.0  # intentional — not a bug


# ---------------------------------------------------------------------------
# F: Music signal
# ---------------------------------------------------------------------------
def music_signal(curr: TemporalFeature, prev: TemporalFeature) -> float:
    """
    Returns 1.0 if the music mood bucket changed between seconds.
    NOTE: music_mood is a coarse heuristic from audio_energy/spectral_flux;
    see adapters/music_mood.py. Not a real music mood classifier.
    """
    return 1.0 if curr.music_mood != prev.music_mood else 0.0
