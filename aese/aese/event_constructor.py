"""
aese/event_constructor.py
Event Constructor — creates Event objects when a boundary is confirmed.

Responsibilities:
  - Maintains an "open" event accumulator between boundaries
  - On confirmed boundary: closes the current event, opens a new one
  - Enforces minimum_event_duration_s=5 (tags short events for merge)
  - Enforces maximum_event_duration_s=300 (force-splits at highest internal fused score)
  - Generates template-based summary (NOT LLM-generated — see README.md)
  - Collects all TemporalFeatures and fused scores in the current event window
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .types import AESEConfig, BoundaryDecision, Event, TemporalFeature

logger = logging.getLogger(__name__)


class EventConstructor:
    """
    Stateful event builder.

    Usage:
        ec = EventConstructor(config, event_embedding_fn, keyframe_fn)
        for tf, decision in stream:
            events = ec.update(tf, decision)
            for event in events:
                yield event
        final = ec.close()  # flush last open event at end-of-stream
        if final:
            yield final
    """

    def __init__(
        self,
        config: AESEConfig,
        event_embedding_fn,  # callable(List[TemporalFeature]) -> np.ndarray
        keyframe_fn,         # callable(List[np.ndarray]) -> Optional[np.ndarray]
    ) -> None:
        self.config = config
        self._embed_fn = event_embedding_fn
        self._keyframe_fn = keyframe_fn
        self._candidate_counter = 0

        # Current open event state
        self._open_start_ms: Optional[float] = None
        self._open_features: List[TemporalFeature] = []
        self._open_fused_scores: List[Tuple[float, float]] = []  # (timestamp_ms, fused_score)
        self._open_decision: Optional[BoundaryDecision] = None

    def update(
        self,
        tf: TemporalFeature,
        decision: BoundaryDecision,
    ) -> List[Event]:
        """
        Process a new TemporalFeature + BoundaryDecision.

        Returns:
            List of completed Events (usually 0 or 1, but may be 2 on a force-split).
        """
        # --- Start a new event if none is open ---
        if self._open_start_ms is None:
            self._open_start_ms = tf.timestamp_ms

        # --- Check for force-split on max duration exceeded ---
        events: List[Event] = []
        duration_s = (tf.timestamp_ms - self._open_start_ms) / 1000.0
        if duration_s > self.config.maximum_event_duration_s:
            logger.info(
                "AESE: force-splitting event at ts=%.0f ms (duration %.1fs > max %.1fs)",
                tf.timestamp_ms,
                duration_s,
                self.config.maximum_event_duration_s,
            )
            # Split at highest internal fused_score point — see event_split.py logic
            self._open_features.append(tf)
            self._open_fused_scores.append((tf.timestamp_ms, decision.fused_score))
            split_events = self._force_split(tf.timestamp_ms)
            events.extend(split_events)
            return events

        # --- Confirmed boundary → close event ---
        if decision.is_boundary:
            rewound_tf = None
            rewound_score = None
            # The motion_spike hard trigger fires on the 2nd consecutive fast_action frame.
            # We rewind the 1st fast_action frame out of the closing event so it belongs to the Action event.
            if decision.dominant_signal == "motion_spike" and len(self._open_features) > 0:
                rewound_tf = self._open_features.pop()
                rewound_score = self._open_fused_scores.pop()

            end_time_ms = rewound_tf.timestamp_ms if rewound_tf else tf.timestamp_ms

            event = self._close_event(
                end_time_ms=end_time_ms,
                confidence=decision.confidence,
                boundary_reason=decision.dominant_signal,
            )
            if event is not None:
                events.append(event)
            
            self._open_start_ms = end_time_ms
            if rewound_tf:
                self._open_features.append(rewound_tf)
                self._open_fused_scores.append(rewound_score)
            
            # The triggering feature belongs to the NEW event
            self._open_features.append(tf)
            self._open_fused_scores.append((tf.timestamp_ms, decision.fused_score))
        else:
            # --- Accumulate current second into open event ---
            self._open_features.append(tf)
            self._open_fused_scores.append((tf.timestamp_ms, decision.fused_score))

        return events

    def close(self) -> Optional[Event]:
        """
        Close the final open event at end-of-stream.
        Returns None if no event is open or if insufficient data.
        """
        if not self._open_features:
            return None
        end_ms = self._open_features[-1].timestamp_ms
        return self._close_event(
            end_time_ms=end_ms,
            confidence=0.5,  # neutral confidence for stream-end boundary
            boundary_reason="stream_end",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_event(
        self,
        end_time_ms: float,
        confidence: float,
        boundary_reason: str,
    ) -> Optional[Event]:
        """Close the current open event window and build an Event object."""
        if not self._open_features or self._open_start_ms is None:
            return None

        start_ms = self._open_start_ms
        duration_ms = end_time_ms - start_ms

        # Minimum duration check — tagged via event_merge later, but log here
        if duration_ms < self.config.minimum_event_duration_s * 1000:
            logger.debug(
                "AESE: event duration %.1f ms < minimum %.1f ms — will be merge candidate",
                duration_ms,
                self.config.minimum_event_duration_s * 1000,
            )

        features = list(self._open_features)

        # Event embedding — mean pool all feature embeddings
        event_emb = self._embed_fn(features)

        # Keyframe selection — use raw images from features (may be None in replay mode)
        raw_images = [
            tf.multimodal_embedding  # proxy if no image; keyframe_fn handles None
            for tf in features
        ]
        key_frame = self._keyframe_fn(features)

        # Importance: derived from peak signals within the event's span
        importance = compute_event_importance(features, boundary_reason)

        # Dominant scene / action labels via majority vote
        scene_label = _majority(tf.scene_label for tf in features)
        action_label = _majority(tf.action_label for tf in features)
        char_counts = [tf.character_count for tf in features]

        # Filter out seconds with no image data (None = not observed)
        observed_counts = [c for c in char_counts if c is not None]
        # character_count_range=None means the entire event had no real image data (not "zero people")
        character_count_range: Optional[List[int]] = sorted(set(observed_counts)) if observed_counts else None
        max_characters_seen: Optional[int] = max(observed_counts) if observed_counts else None
        character_data_available = any(getattr(tf, "image_available", True) for tf in features)

        char_count_max = max_characters_seen

        # Template-based summary — NOT LLM-generated (see README.md)
        # Attempt VLM-generated summary first if a real image is available.
        # If VLM fails/unavailable, returns empty string, and pipeline.py will
        # backfill it with build_template_summary using the final event_type.
        summary = _make_vlm_or_template_summary(
            features=features,
            scene_label=scene_label,
            action_label=action_label,
        ) or ""

        # Location label from dominant scene label
        location_label = scene_label if scene_label != "unknown" else None

        event = Event(
            event_id=self._candidate_counter,
            start_time_ms=start_ms,
            end_time_ms=end_time_ms,
            duration_ms=duration_ms,
            event_embedding=event_emb,
            importance=importance,
            confidence=confidence,
            summary=summary,
            boundary_reason=boundary_reason,
            event_type="Scene",  # placeholder — will be overwritten by EventClassifier
            key_frame=key_frame,
            character_count_range=character_count_range,
            max_characters_seen=max_characters_seen,
            character_data_available=character_data_available,
            location_label=location_label,
        )

        self._candidate_counter += 1

        # Reset open event state
        self._open_start_ms = None
        self._open_features = []
        self._open_fused_scores = []
        self._open_decision = None

        return event

    def _force_split(self, current_ts_ms: float) -> List[Event]:
        """
        Force-split the current event at the timestamp with the highest fused_score.
        Used when event exceeds maximum_event_duration_s.
        See event_split.py for the standalone logic.
        """
        events: List[Event] = []

        if not self._open_fused_scores or not self._open_features:
            return events

        # Find split point: timestamp with locally highest fused_score
        split_ts = max(self._open_fused_scores, key=lambda x: x[1])[0]
        # Avoid splitting at the very start (degenerate)
        if split_ts == self._open_start_ms:
            if len(self._open_fused_scores) > 1:
                split_ts = self._open_fused_scores[len(self._open_fused_scores) // 2][0]

        # Partition features into pre-split and post-split
        pre_features = [tf for tf in self._open_features if tf.timestamp_ms <= split_ts]
        post_features = [tf for tf in self._open_features if tf.timestamp_ms > split_ts]

        if pre_features:
            saved_features = self._open_features
            saved_scores = self._open_fused_scores
            self._open_features = pre_features
            self._open_fused_scores = [(ts, s) for ts, s in saved_scores if ts <= split_ts]
            ev1 = self._close_event(
                end_time_ms=split_ts,
                confidence=0.6,
                boundary_reason="force_split_max_duration",
            )
            if ev1:
                events.append(ev1)
            # Reopen with remaining features
            self._open_start_ms = post_features[0].timestamp_ms if post_features else None
            self._open_features = post_features
            self._open_fused_scores = [(ts, s) for ts, s in saved_scores if ts > split_ts]

        return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _majority(iterable) -> str:
    """Return most common string from iterable."""
    from collections import Counter
    counts = Counter(iterable)
    return counts.most_common(1)[0][0] if counts else "unknown"


def compute_event_importance(features: List[TemporalFeature], boundary_reason: str) -> float:
    """
    Importance describes how significant THIS event's content is, not how the
    boundary that created it was detected. Use peak (not mean) motion/novelty/audio
    within the event's own span -- a short intense event shouldn't be penalized by
    dividing its peak by its own (possibly short) duration.
    """
    peak_motion = max((f.motion_score for f in features), default=0.0)
    peak_novelty = max((f.novelty_score for f in features), default=0.0)
    mean_audio = sum(f.audio_energy for f in features) / len(features) if features else 0.0
    dialogue_density = sum(f.dialogue_present for f in features) / len(features) if features else 0.0

    importance = (
        0.40 * peak_motion +
        0.25 * peak_novelty +
        0.20 * mean_audio +
        0.15 * dialogue_density
    )
    return min(max(importance, 0.0), 1.0)


def build_template_summary(event_type: str, scene_label: str, max_characters_seen: Optional[int]) -> str:
    """
    Template-based event summary.
    MUST use the same aggregated max_characters_seen value as the JSON field.
    Format: "<event_type> event in <scene>, <n> people present"
    """
    if max_characters_seen is None:
        people_str = "character data unavailable"
    elif max_characters_seen == 0:
        people_str = "no people detected"
    else:
        people_str = f"{max_characters_seen} {'person' if max_characters_seen == 1 else 'people'} present"

    return f"{event_type} event in {scene_label}, {people_str}"


def _make_vlm_or_template_summary(
    features: list,
    scene_label: str,
    action_label: str,

) -> str:
    """
    Generate an event summary using FastVLM if a real image is available,
    otherwise fall back to the template-based _make_summary.

    VLM path:
      - Selects the feature with the lowest motion_score (sharpest frame)
        that also has a representative_image (real pixel data).
      - Calls adapters.fastvlm.caption_event with the image + context.
      - Returns the VLM caption if non-empty.

    Fallback (no image / VLM unavailable):
      - Returns the existing template-based summary string.
    """
    # Find the sharpest feature with a real image
    candidates = [
        tf for tf in features
        if getattr(tf, "representative_image", None) is not None
    ]
    if candidates:
        keyframe_feature = min(candidates, key=lambda tf: tf.motion_score)
        image = keyframe_feature.representative_image
        dialogue_text = next(
            (tf.dialogue_text for tf in reversed(features) if tf.dialogue_text),
            None,
        )
        try:
            from .adapters.fastvlm import caption_event
            vlm_summary = caption_event(image, scene_label, action_label, dialogue_text)
            if vlm_summary:
                logger.debug(
                    "AESE event_constructor: VLM summary generated (%d chars)", len(vlm_summary)
                )
                return vlm_summary
        except Exception as exc:
            logger.debug(
                "AESE event_constructor: VLM summary failed (%s) — using template", exc
            )

    # Fallback: no summary generated here, pipeline.py will fill it in
    return None
