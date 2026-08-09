"""
aese/boundary/confidence.py
Boundary confidence scoring — §5.7, §16.

Confidence is separate from fused_score:
  - A score well above threshold (e.g. 0.95) is high confidence → decide immediately.
  - A score near threshold (±margin) is low confidence → hold for up to 2 more seconds
    before committing.

This implements the "2-second confidence hold" non-functional requirement:
  "if fused_score is within a margin (e.g. ±0.05) of boundary_threshold, hold for
   up to 2 more seconds and re-evaluate with fresh signals before committing."
"""
from __future__ import annotations


def compute_confidence(fused_score: float, threshold: float, margin: float = 0.05) -> float:
    """
    Compute confidence of a boundary decision from the fused score.

    Confidence formula:
      - score > threshold + margin     → high confidence: normalize to [0.8, 1.0]
      - score within ±margin           → low confidence: [0.4, 0.8]
      - score < threshold - margin     → not a boundary: [0.0, 0.4]

    Args:
        fused_score: The fused boundary score in [0, 1].
        threshold: Boundary threshold from AESEConfig.
        margin: Half-width of the "low-confidence zone" around the threshold.

    Returns:
        float: Confidence in [0, 1].
    """
    if fused_score >= threshold + margin:
        # Well above threshold — scale [threshold+margin, 1.0] → [0.8, 1.0]
        excess = fused_score - (threshold + margin)
        max_excess = 1.0 - (threshold + margin)
        if max_excess <= 0:
            return 1.0
        return 0.8 + 0.2 * min(excess / max_excess, 1.0)

    elif fused_score >= threshold - margin:
        # Near threshold — low confidence zone
        # Scale [threshold-margin, threshold+margin] → [0.4, 0.8]
        pos = fused_score - (threshold - margin)
        zone_width = 2 * margin
        if zone_width <= 0:
            return 0.5
        return 0.4 + 0.4 * min(pos / zone_width, 1.0)

    else:
        # Below threshold
        # Scale [0, threshold-margin] → [0.0, 0.4]
        max_val = threshold - margin
        if max_val <= 0:
            return 0.0
        return 0.4 * min(fused_score / max_val, 1.0)


def is_high_confidence(confidence: float, min_confidence: float = 0.75) -> bool:
    """Returns True if confidence meets the threshold for immediate decision."""
    return confidence >= min_confidence
