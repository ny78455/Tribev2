"""
aese/adapters/music_mood.py
Music mood adapter.

# STUB: Heuristic bucket from audio energy + spectral flux into mood categories.
NOT a music genre or mood classifier. No trained model is used — only rule-based
thresholds on energy and spectral flux, clearly labeled as heuristic.

Mood buckets:
  "calm"      — low energy (< 0.08), low flux
  "tense"     — moderate energy (0.08–0.25) or high flux
  "energetic" — high energy (≥ 0.25)

Spectral flux estimation: Since Module 1 does not expose spectral flux directly,
it is estimated from the per-frame delta in audio_energy (rolling). This is a
very coarse proxy. See DECISIONS.md §6.

Future work: replace with a lightweight trained music mood classifier
(e.g. musicnn, MusicCNN, or a fine-tuned YAMNet).
"""
from __future__ import annotations

from typing import Optional


# Energy thresholds for mood bucketing
_CALM_ENERGY_MAX = 0.08
_ENERGETIC_ENERGY_MIN = 0.25

# Spectral flux thresholds (estimated from audio_energy delta)
_HIGH_FLUX_THRESHOLD = 0.08


def label_mood(audio_energy: float, spectral_flux: float = 0.0) -> str:
    """
    # STUB: Heuristic music mood label from energy and spectral flux.

    Args:
        audio_energy: float in [0, 1], from FramePacket.audio_energy.
        spectral_flux: float ≥ 0.0, estimated from energy delta (Module 1 does not provide this).

    Returns:
        str: One of "calm" | "tense" | "energetic".
    """
    # High energy → energetic regardless of flux
    if audio_energy >= _ENERGETIC_ENERGY_MIN:
        return "energetic"

    # Low energy + low flux → calm
    if audio_energy < _CALM_ENERGY_MAX and spectral_flux < _HIGH_FLUX_THRESHOLD:
        return "calm"

    # Mid-range energy or high flux → tense
    return "tense"


def estimate_spectral_flux(prev_audio_energy: Optional[float], curr_audio_energy: float) -> float:
    """
    Estimate spectral flux as the absolute delta in audio_energy between frames.
    # STUB: This is not a real spectral flux measure (which requires FFT of audio frames).
    Used only because Module 1 does not expose spectral flux directly.
    See DECISIONS.md §6.

    Args:
        prev_audio_energy: Previous frame's audio_energy, or None for the first frame.
        curr_audio_energy: Current frame's audio_energy.

    Returns:
        float: Estimated flux value ≥ 0.0.
    """
    if prev_audio_energy is None:
        return 0.0
    return abs(curr_audio_energy - prev_audio_energy)
