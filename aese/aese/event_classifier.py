"""
aese/event_classifier.py
Event Type Classifier — §5.14, §23.

Rule-based classifier over aggregated TemporalFeature labels.
NOT a trained model — intentional heuristic V1.

Event type categories:
  "Dialogue"    — majority of seconds have dialogue_present=True
  "Action"      — majority of seconds have action_label="fast_action"
  "Transition"  — short event (<15s) OR camera_cue present in most seconds
  "Scene"       — default catch-all (establishing shots, no clear activity)

Rules are applied in priority order:
  1. Dialogue (highest priority — dialogue is the strongest categorical signal)
  2. Action (fast motion majority)
  3. Transition (short duration or camera-cue dominated)
  4. Scene (default)

Document this ruleset — marked as heuristic V1.
Future work: replace with a multi-class learned classifier over the event embedding.
"""
from __future__ import annotations

import logging
from typing import List

from .types import AESEConfig, Event, TemporalFeature

logger = logging.getLogger(__name__)

# Classification thresholds
_DIALOGUE_MAJORITY_THRESHOLD = 0.4     # ≥40% of seconds with dialogue → Dialogue
_ACTION_MAJORITY_THRESHOLD = 0.4       # ≥40% of seconds with fast_action → Action
_TRANSITION_MAX_DURATION_MS = 8000.0   # Events < 8s are likely transitions
_CAMERA_CUE_THRESHOLD = 0.5           # ≥50% of seconds with camera cue → Transition


def classify_event(
    event: Event,
    features: List[TemporalFeature],
) -> str:
    """
    Rule-based event type classification.

    Args:
        event: The constructed Event (with duration, start/end times).
        features: TemporalFeatures composing this event.

    Returns:
        str: One of "Dialogue" | "Action" | "Transition" | "Scene".
    """
    if not features:
        return "Scene"

    n = len(features)

    # --- Rule 1: Dialogue --- (highest priority)
    dialogue_frac = sum(1 for tf in features if tf.dialogue_present) / n
    if dialogue_frac >= _DIALOGUE_MAJORITY_THRESHOLD:
        logger.debug(
            "AESE classifier: event %d → Dialogue (dialogue_frac=%.2f)",
            event.event_id, dialogue_frac,
        )
        return "Dialogue"

    # --- Rule 2: Action ---
    action_frac = sum(1 for tf in features if tf.action_label == "fast_action") / n
    if action_frac >= _ACTION_MAJORITY_THRESHOLD:
        logger.debug(
            "AESE classifier: event %d → Action (action_frac=%.2f)",
            event.event_id, action_frac,
        )
        return "Action"

    # --- Rule 3: Transition ---
    is_short = event.duration_ms < _TRANSITION_MAX_DURATION_MS
    camera_frac = sum(
        1 for tf in features if tf.camera_cue in ("cut", "fade", "black")
    ) / n
    if is_short or camera_frac >= _CAMERA_CUE_THRESHOLD:
        logger.debug(
            "AESE classifier: event %d → Transition (short=%s, camera_frac=%.2f)",
            event.event_id, is_short, camera_frac,
        )
        return "Transition"

    # --- Rule 4: Scene (default) ---
    logger.debug("AESE classifier: event %d → Scene (default)", event.event_id)
    return "Scene"
