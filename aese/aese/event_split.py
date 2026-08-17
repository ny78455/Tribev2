"""
aese/event_split.py
Event Split — §5.13, §22.

Force-split events that exceed maximum_event_duration_s (300s) at the timestamp
with the locally highest fused_score observed during the event's span.

Even if that score never crossed boundary_threshold, we must split to prevent
runaway events. This is a safety valve for very long or static scenes.

Note: The EventConstructor already handles in-line force-splitting during event
accumulation. This module provides the standalone split logic that can also be
applied post-hoc to already-closed events.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .event_embedding import pool_event_embedding
from .event_constructor import _make_summary, _majority
from .types import AESEConfig, Event, TemporalFeature

logger = logging.getLogger(__name__)


def should_split(event: Event, max_duration_s: float = 300.0) -> bool:
    """Return True if the event exceeds the maximum duration."""
    return event.duration_ms > max_duration_s * 1000.0


def split_event(
    event: Event,
    features: List[TemporalFeature],
    fused_scores: List[Tuple[float, float]],  # (timestamp_ms, fused_score)
    config: AESEConfig,
    next_event_id: int,
) -> List[Event]:
    """
    Force-split an event at the timestamp with the highest internal fused_score.

    Args:
        event: The oversized Event to split.
        features: All TemporalFeatures that compose this event.
        fused_scores: List of (timestamp_ms, fused_score) for each second in the event.
        config: AESEConfig (for duration limits).
        next_event_id: ID to assign to the second split event.

    Returns:
        List of 2 Events (or 1 if the split point is degenerate).
    """
    if not features or not fused_scores:
        return [event]

    # Find split point: timestamp with highest fused_score
    split_ts, peak_score = max(fused_scores, key=lambda x: x[1])

    # Avoid degenerate split at the very first or last second
    if split_ts == event.start_time_ms and len(fused_scores) > 2:
        fused_scores_sorted = sorted(fused_scores, key=lambda x: x[1], reverse=True)
        for ts, score in fused_scores_sorted:
            if ts != event.start_time_ms:
                split_ts = ts
                break

    if split_ts == event.start_time_ms or split_ts == event.end_time_ms:
        logger.warning(
            "AESE split: degenerate split point for event %d — no split performed",
            event.event_id,
        )
        return [event]

    # Partition features
    pre_features = [tf for tf in features if tf.timestamp_ms <= split_ts]
    post_features = [tf for tf in features if tf.timestamp_ms > split_ts]

    if not pre_features or not post_features:
        return [event]

    logger.info(
        "AESE force-split event %d (%.1fs) at ts=%.0f ms (peak_score=%.3f)",
        event.event_id,
        event.duration_ms / 1000,
        split_ts,
        peak_score,
    )

    # Build first event (pre-split)
    pre_emb = pool_event_embedding(pre_features)
    pre_scene = _majority(tf.scene_label for tf in pre_features)
    pre_action = _majority(tf.action_label for tf in pre_features)
    pre_observed = [tf.character_count for tf in pre_features if tf.character_count is not None]
    pre_char_range = sorted(set(pre_observed)) if pre_observed else None
    pre_char_max = max(pre_observed) if pre_observed else None
    event1 = Event(
        event_id=event.event_id,
        start_time_ms=event.start_time_ms,
        end_time_ms=split_ts,
        duration_ms=split_ts - event.start_time_ms,
        event_embedding=pre_emb,
        importance=float(np.mean([tf.novelty_score for tf in pre_features])),
        confidence=event.confidence,
        summary=_make_summary(pre_scene, pre_action, pre_char_max),
        boundary_reason="force_split_max_duration",
        event_type=event.event_type,
        key_frame=event.key_frame,
        character_count_range=pre_char_range,
        max_characters_seen=pre_char_max,
        location_label=pre_scene if pre_scene != "unknown" else None,
    )

    # Build second event (post-split)
    post_emb = pool_event_embedding(post_features)
    post_scene = _majority(tf.scene_label for tf in post_features)
    post_action = _majority(tf.action_label for tf in post_features)
    post_observed = [tf.character_count for tf in post_features if tf.character_count is not None]
    post_char_range = sorted(set(post_observed)) if post_observed else None
    post_char_max = max(post_observed) if post_observed else None
    event2 = Event(
        event_id=next_event_id,
        start_time_ms=split_ts,
        end_time_ms=event.end_time_ms,
        duration_ms=event.end_time_ms - split_ts,
        event_embedding=post_emb,
        importance=float(np.mean([tf.novelty_score for tf in post_features])),
        confidence=event.confidence,
        summary=_make_summary(post_scene, post_action, post_char_max),
        boundary_reason="force_split_continuation",
        event_type=event.event_type,
        key_frame=None,
        character_count_range=post_char_range,
        max_characters_seen=post_char_max,
        location_label=post_scene if post_scene != "unknown" else None,
    )

    return [event1, event2]
