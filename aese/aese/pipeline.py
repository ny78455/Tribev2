"""
aese/pipeline.py
Pipeline Orchestrator -- wires all AESE modules together.

Streaming generator:
    run(frame_packet_stream, config) -> Iterator[Event]

Architecture:
    FramePackets -> FeatureAggregator -> ContextBuffer
                -> CandidateDetector (signals + fusion + confidence)
                -> EventConstructor (min/max duration)
                -> OnlineMerger
                -> EventClassifier
                -> generate_summary()   <-- ONCE per finalized event, off the hot path
                -> yield Event

Non-functional requirements (§8):
    - Latency: <100ms per boundary decision (logged)
      NOTE: generate_summary() is intentionally NOT inside this loop.
    - Memory: rolling buffer -- never holds full movie in RAM
    - Max delay: 2s hold for confidence (enforced in CandidateDetector)
    - Streaming: no future frames peeked -- pure online processing

Modes:
    - Live mode: receives FramePacket objects with .image arrays (from ASVL)
    - Manifest-replay mode: receives FramePackets from JSON manifest (no .image)
      In replay mode, image-dependent adapters receive a black placeholder frame.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterator, List, Optional

import numpy as np

from .adapters.character_cluster import CharacterClusterer, get_character_labels_for_event
from .aggregator import FeatureAggregator
from .boundary.candidate_detector import CandidateDetector
from .context_buffer import ContextBuffer
from .event_classifier import classify_event
from .event_constructor import EventConstructor, build_template_summary
from .event_embedding import pool_event_embedding
from .event_graph import EventGraph
from .event_merge import OnlineMerger
from .keyframe import select_keyframe
from .summary import generate_summary, _summary_call_counter
from .types import AESEConfig, Event, FramePacket

logger = logging.getLogger(__name__)

_DECISION_LATENCY_WARN_MS = 100.0
_MIN_FEATURES_FOR_BOUNDARY = 2  # need at least 2 seconds to make a boundary decision


def _get_rss_mb() -> float:
    """Return current RSS in MB (cross-platform via psutil)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def _finalize_event(
    event: Event,
    next_id: int,
    clusterer: Optional["CharacterClusterer"] = None,
    event_features_map: Optional[dict] = None,
) -> None:
    """
    Assign contiguous ID, assign character labels, and generate the final summary.

    Summary generation order:
      1. generate_summary() -- calls VLM if available + real image, with validation.
         Internally falls back to build_template_summary() on any failure.
      2. If event.summary is still empty (e.g. generate_summary raised unexpectedly),
         build_template_summary() is the guaranteed safety net.

    Character labeling:
      - If clusterer is provided, resolves face embeddings from the event's source
        TemporalFeatures into anonymous labels ("Person A", "Person B", ...).
      - Empty list if clustering is unavailable (manifest-replay mode, no --video).

    This function runs AFTER merge/classify -- once per finalized event, never
    in the per-second boundary-detection loop.
    """
    event.event_id = next_id

    # --- Character clustering (Fix 4) ---
    if clusterer is not None and event_features_map is not None:
        feats = event_features_map.get(event.event_id, [])
        face_embeddings_per_second = [getattr(tf, "face_embeddings", []) for tf in feats]
        event.character_labels = get_character_labels_for_event(face_embeddings_per_second, clusterer)

    # --- Summary generation ---
    keyframe_image = None
    if event.key_frame is not None and hasattr(event.key_frame, 'shape') and event.key_frame.ndim == 3:
        # key_frame is a real pixel array (live mode) -- use it for VLM summary
        keyframe_image = event.key_frame
    event.summary = generate_summary(event, keyframe_image)
    if not event.summary:
        # Guaranteed safety net -- should never be reached since generate_summary always returns str
        event.summary = build_template_summary(
            event_type=event.event_type,
            scene_label=event.location_label or "unknown",
            max_characters_seen=event.max_characters_seen,
        )


def run(
    frame_packet_stream: Iterator[FramePacket],
    config: AESEConfig,
) -> Iterator[Event]:
    """
    Main AESE pipeline: consume FramePackets, yield completed Events online.

    Args:
        frame_packet_stream: Iterator of FramePacket objects (from ASVL or manifest replay).
        config: AESEConfig instance (use aese.config.load_config() to create).

    Yields:
        Event objects, one at a time, as they are completed (online, no buffering).
    """
    # --- Initialize subsystems ---
    aggregator = FeatureAggregator(config)
    buffer = ContextBuffer(buffer_seconds=config.buffer_seconds)
    detector = CandidateDetector(config, buffer)
    constructor = EventConstructor(
        config=config,
        event_embedding_fn=pool_event_embedding,
        keyframe_fn=lambda features: select_keyframe(features, strategy="lowest_blur"),
    )
    merger = OnlineMerger(config)
    event_graph = EventGraph()
    clusterer = CharacterClusterer()  # Fix 4: one per video run, not per event

    prev_feature = None
    decision_latencies_ms: List[float] = []
    peak_rss_mb = 0.0
    packets_processed = 0
    events_emitted = 0
    # Track features per event for classifier
    event_features: dict = {}  # event_id → List[TemporalFeature]
    current_event_features: List = []

    logger.info(
        "AESE: pipeline started (buffer=%.0fs, threshold=%.2f, max_event_duration=%.0fs). "
        "Hard-trigger boundaries: camera_cut=0.95, action_transition=0.85. "
        "Force-split safety net fires at %.0fs.",
        config.buffer_seconds,
        config.boundary_threshold,
        config.maximum_event_duration_s,
        config.maximum_event_duration_s,
    )


    for fp in frame_packet_stream:
        loop_start = time.perf_counter()
        packets_processed += 1

        # --- Aggregate into per-second TemporalFeatures ---
        new_features = aggregator.push_all(fp)

        for tf in new_features:
            # Push into context buffer
            buffer.push(tf)
            current_event_features.append(tf)

            # --- Boundary detection ---
            decision = detector.update(tf, prev_feature)

            # --- Event construction ---
            completed_events = constructor.update(tf, decision)

            for event in completed_events:
                # Save the features that composed this event for classifier
                event_features[event.event_id] = list(current_event_features[:-1] or current_event_features)
                current_event_features = [tf]  # reset; current tf starts new event

                # --- Classify event type ---
                ev_feats = event_features.get(event.event_id, [tf])
                event.event_type = classify_event(event, ev_feats)

                # --- Online merge ---
                finalized = merger.process(event)
                if finalized is not None:
                    _finalize_event(finalized, events_emitted, clusterer, event_features)
                    event_graph.add_event(finalized)
                    buffer.record_boundary(finalized.end_time_ms)
                    events_emitted += 1
                    yield finalized

            prev_feature = tf

        # --- Latency logging ---
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        decision_latencies_ms.append(loop_ms)
        if loop_ms > _DECISION_LATENCY_WARN_MS:
            logger.warning(
                "AESE: decision loop %.1f ms > 100ms threshold at ts=%.1f ms",
                loop_ms, fp.timestamp_ms,
            )

        # Peak RSS sampling (every 50 packets)
        if packets_processed % 50 == 0:
            rss = _get_rss_mb()
            if rss > peak_rss_mb:
                peak_rss_mb = rss

    # --- Flush aggregator trailing partial second ---
    for tf in aggregator.flush():
        buffer.push(tf)
        current_event_features.append(tf)
        decision = detector.update(tf, prev_feature)
        completed_events = constructor.update(tf, decision)
        for event in completed_events:
            ev_feats = event_features.get(event.event_id, current_event_features)
            event.event_type = classify_event(event, ev_feats)
            finalized = merger.process(event)
            if finalized is not None:
                _finalize_event(finalized, events_emitted, clusterer, event_features)
                event_graph.add_event(finalized)
                buffer.record_boundary(finalized.end_time_ms)
                events_emitted += 1
                yield finalized
        prev_feature = tf

    # --- Close final open event ---
    final_event = constructor.close()
    if final_event is not None:
        ev_feats = current_event_features if current_event_features else []
        final_event.event_type = classify_event(final_event, ev_feats)
        finalized = merger.process(final_event)
        if finalized is not None:
            _finalize_event(finalized, events_emitted, clusterer, event_features)
            event_graph.add_event(finalized)
            events_emitted += 1
            yield finalized

    # Flush merger's held event
    last_held = merger.finalize()
    if last_held is not None:
        last_held.event_type = classify_event(last_held, [])
        _finalize_event(last_held, events_emitted, clusterer, event_features)
        event_graph.add_event(last_held)
        events_emitted += 1
        yield last_held

    # --- End-of-run stats ---
    if decision_latencies_ms:
        sorted_lat = sorted(decision_latencies_ms)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        logger.info(
            "AESE run complete: %d packets processed, %d events emitted | "
            "p95 latency=%.1f ms | peak RSS=%.1f MB",
            packets_processed, events_emitted, p95, peak_rss_mb,
        )
        if p95 > _DECISION_LATENCY_WARN_MS:
            logger.warning("AESE: p95 boundary-decision latency %.1f ms > 100ms target.", p95)
        if peak_rss_mb > 1024:
            logger.warning("AESE: peak RSS %.1f MB > 1GB target.", peak_rss_mb)
