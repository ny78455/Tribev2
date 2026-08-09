"""
aese/pipeline.py
Pipeline Orchestrator — wires all AESE modules together.

Streaming generator:
    run(frame_packet_stream, config) -> Iterator[Event]

Architecture:
    FramePackets → FeatureAggregator → ContextBuffer
                → CandidateDetector (signals + fusion + confidence)
                → EventConstructor (min/max duration)
                → OnlineMerger
                → EventClassifier
                → yield Event

Non-functional requirements (§8):
    - Latency: <100ms per boundary decision (logged)
    - Memory: rolling buffer — never holds full movie in RAM
    - Max delay: 2s hold for confidence (enforced in CandidateDetector)
    - Streaming: no future frames peeked — pure online processing

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

from .aggregator import FeatureAggregator
from .boundary.candidate_detector import CandidateDetector
from .context_buffer import ContextBuffer
from .event_classifier import classify_event
from .event_constructor import EventConstructor
from .event_embedding import pool_event_embedding
from .event_graph import EventGraph
from .event_merge import OnlineMerger
from .keyframe import select_keyframe
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

    prev_feature = None
    decision_latencies_ms: List[float] = []
    peak_rss_mb = 0.0
    packets_processed = 0
    events_emitted = 0
    # Track features per event for classifier
    event_features: dict = {}  # event_id → List[TemporalFeature]
    current_event_features: List = []

    logger.info("AESE: pipeline started (buffer=%.0fs, threshold=%.2f)",
                config.buffer_seconds, config.boundary_threshold)

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
            event_graph.add_event(finalized)
            events_emitted += 1
            yield finalized

    # Flush merger's held event
    last_held = merger.finalize()
    if last_held is not None:
        last_held.event_type = classify_event(last_held, [])
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
