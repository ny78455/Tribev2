"""
aese/event_merge.py
Event Merge — §5.12, §21.

After event construction, check adjacent events pairwise.
Merge criteria (all three must be true):
  1. same_chars: same set of character counts (stub proxy for same cast)
  2. same_location: same location_label
  3. low_distance: embedding distance < (1 - merge_threshold)

The merge decision happens online — we check the just-closed event against the
one before it. This respects the 2-second max-delay budget (no indefinite deferral).

Post-merge: update start/end times, re-pool embedding, update summary.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .boundary.embedding_change import embedding_distance
from .event_embedding import pool_event_embedding
from .event_constructor import _make_summary, _majority
from .types import AESEConfig, Event

logger = logging.getLogger(__name__)


def should_merge(a: Event, b: Event, merge_threshold: float) -> bool:
    """
    Return True if events a and b should be merged.

    Criteria (all must pass):
      1. Same character count set (stub proxy for same characters present)
      2. Same location label (or both None)
      3. Embedding distance < (1 - merge_threshold)

    Args:
        a: Earlier event.
        b: Later event (immediately follows a).
        merge_threshold: From AESEConfig.merge_threshold (default 0.80).

    Returns:
        bool: True if events should be merged into one.
    """
    # 1. Same characters (by count set — stub; no real identity)
    same_chars = set(a.characters) == set(b.characters)

    # 2. Same location
    same_location = a.location_label == b.location_label

    # 3. Embedding distance threshold
    dist = embedding_distance(a.event_embedding, b.event_embedding, metric="cosine")
    low_distance = dist < (1.0 - merge_threshold)

    result = same_chars and same_location and low_distance
    if result:
        logger.debug(
            "AESE merge: event %d + %d (dist=%.3f, chars=%s, loc=%s)",
            a.event_id, b.event_id, dist, same_chars, same_location,
        )
    return result


def merge_events(a: Event, b: Event) -> Event:
    """
    Merge two adjacent events into one combined event.
    Uses the earlier event's ID and start time; the later event's end time.
    Re-pools embedding as mean of the two embeddings (cheap approximation).
    """
    merged_emb = ((a.event_embedding.astype(np.float64)
                   + b.event_embedding.astype(np.float64)) / 2.0).astype(np.float32)

    duration_ms = b.end_time_ms - a.start_time_ms

    # Merge character lists
    merged_chars = list(set(a.characters) | set(b.characters))

    # Location: use a's location if they agree (should be same)
    location = a.location_label

    # Importance: weighted mean by duration
    total_dur = a.duration_ms + b.duration_ms
    if total_dur > 0:
        importance = (a.importance * a.duration_ms + b.importance * b.duration_ms) / total_dur
    else:
        importance = (a.importance + b.importance) / 2.0

    # Confidence: min of the two (conservative)
    confidence = min(a.confidence, b.confidence)

    # Summary: re-generate with merged info
    char_count_max = max(merged_chars) if merged_chars else 0
    scene_label = location if location else "unknown"
    action_label = a.summary.split(" ")[0].lower() if a.summary else "scene"
    summary = _make_summary(scene_label, action_label, char_count_max)

    # Keyframe: prefer the one with higher novelty (take a's keyframe for simplicity)
    key_frame = a.key_frame if a.key_frame is not None else b.key_frame

    return Event(
        event_id=a.event_id,  # preserve earlier ID
        start_time_ms=a.start_time_ms,
        end_time_ms=b.end_time_ms,
        duration_ms=duration_ms,
        event_embedding=merged_emb,
        importance=float(importance),
        confidence=float(confidence),
        summary=summary,
        boundary_reason=b.boundary_reason,  # use the closing boundary's reason
        event_type=a.event_type,  # preserve earlier type; classifier will overwrite
        key_frame=key_frame,
        characters=merged_chars,
        location_label=location,
    )


class OnlineMerger:
    """
    Online event merger — checks each newly completed event against the last
    and merges if the merge criteria pass.

    Operates within the 2-second max-delay budget: merge decisions are made
    immediately when a new event is closed, not deferred.
    """

    def __init__(self, config: AESEConfig) -> None:
        self.config = config
        self._last_event: Optional[Event] = None

    def process(self, event: Event) -> Optional[Event]:
        """
        Consider merging the incoming event with the previous one.

        Returns:
            - None if the event was absorbed into the previous (merge in progress)
            - The previous event if it was finalized (not merged)
            (The new event is held pending the next event's arrival.)

        Usage: call finalize() after the last event to flush the held event.
        """
        if self._last_event is None:
            # First event — just hold it
            self._last_event = event
            return None

        # Check min duration before merge (short events are merge candidates)
        both_short = (
            self._last_event.duration_ms < self.config.minimum_event_duration_s * 1000
            or event.duration_ms < self.config.minimum_event_duration_s * 1000
        )
        try_merge = both_short or should_merge(
            self._last_event, event, self.config.merge_threshold
        )

        if try_merge:
            # Merge into running event
            self._last_event = merge_events(self._last_event, event)
            return None
        else:
            # Finalize the previous event, hold the new one
            finalized = self._last_event
            self._last_event = event
            return finalized

    def finalize(self) -> Optional[Event]:
        """Flush the held event at end-of-stream."""
        event = self._last_event
        self._last_event = None
        return event
