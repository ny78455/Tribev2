"""
tests/test_performance_gates.py
Automated performance gates for the AESE pipeline.

These tests enforce the non-functional requirements from Section 8 of the
original Module 2 contract. They exist specifically to catch the class of
regression where a slow call (e.g. a generative VLM) is silently introduced
into the per-second boundary-detection hot path.

History:
  - 2026-08-18: Added after the VLM-in-hot-path regression that caused a
    12-minute runtime on an 81s clip (720s actual vs 90s budget).

CI requirement: both tests must pass on every commit. A failure here means
a slow call has re-entered the hot path.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Iterator, List

import numpy as np
import pytest

from aese.types import AESEConfig, FramePacket


# ---------------------------------------------------------------------------
# Shared synthetic fight-clip stream (same as test_regression_fight_clip.py)
# ---------------------------------------------------------------------------

def _make_packet(
    frame_id: int,
    timestamp_ms: float,
    motion_score: float,
    scene_change: bool,
    audio_energy: float = 0.05,
    novelty_score: float = 0.15,
) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        fps_used=1.0,
        motion_score=motion_score,
        scene_change=scene_change,
        audio_energy=audio_energy,
        novelty_score=novelty_score,
        decision_reason="perf_test",
        subtitle_text=None,
        image=None,
    )


def _fight_clip_stream() -> Iterator[FramePacket]:
    packets: List[FramePacket] = []
    for s in range(30):
        packets.append(_make_packet(s, float(s * 1000), 0.05, False, 0.10, 0.10))
    for s in range(30, 50):
        packets.append(_make_packet(s, float(s * 1000), 0.70, False, 0.40, 0.55))
    for s in range(50, 69):
        packets.append(_make_packet(s, float(s * 1000), 0.08, False, 0.08, 0.12))
    packets.append(_make_packet(69, 69000.0, 0.03, True, 0.20, 0.45))
    for s in range(70, 81):
        packets.append(_make_packet(s, float(s * 1000), 0.02, False, 0.05, 0.08))
    return iter(packets)


# ---------------------------------------------------------------------------
# Test 1 -- Per-second decision latency gate
# ---------------------------------------------------------------------------

def test_per_second_decision_latency():
    """
    No per-second boundary decision may exceed 100ms at p95.

    This is the exact latency requirement from Section 8 of the AESE contract.
    This test guards against the VLM-in-hot-path regression: a generative VLM
    call inside the boundary loop caused 12-minute runtimes on short clips.

    Method: measure wall-clock time for each push_all() + detector.update()
    cycle (the inner loop in pipeline.run()), collect all latencies, assert p95 < 100ms.
    """
    from aese.aggregator import FeatureAggregator
    from aese.boundary.candidate_detector import CandidateDetector
    from aese.context_buffer import ContextBuffer

    config = AESEConfig()
    aggregator = FeatureAggregator(config)
    buffer = ContextBuffer(buffer_seconds=config.buffer_seconds)
    detector = CandidateDetector(config, buffer)
    prev_feature = None
    latencies_ms: List[float] = []

    for fp in _fight_clip_stream():
        t0 = time.perf_counter()
        new_features = aggregator.push_all(fp)
        for tf in new_features:
            buffer.push(tf)
            detector.update(tf, prev_feature)
            prev_feature = tf
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    for tf in aggregator.flush():
        t0 = time.perf_counter()
        buffer.push(tf)
        detector.update(tf, prev_feature)
        prev_feature = tf
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    assert latencies_ms, "No latency samples collected -- check stream generation"

    sorted_lat = sorted(latencies_ms)
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
    p95 = sorted_lat[p95_idx]

    assert p95 < 100.0, (
        f"p95 per-second decision time is {p95:.1f}ms, must be <100ms. "
        f"A slow call (e.g. a generative VLM) has re-entered the hot path. "
        f"All latencies (ms): {[f'{x:.1f}' for x in sorted_lat]}"
    )


# ---------------------------------------------------------------------------
# Test 2 -- Full pipeline runtime budget
# ---------------------------------------------------------------------------

def test_full_pipeline_runtime_budget():
    """
    Full pipeline on the 81s synthetic fight clip must complete in <90s wall-clock.

    The regression this guards against: a generative VLM (~5-8s per call) was
    called ~81 times inside the hot path, causing ~720s total runtime.
    With the VLM moved to post-finalization (once per finalized event, ~8 events),
    the pipeline must run comfortably within real-time on modern hardware.

    Budget: 90s for an 81s clip (1.1x real-time headroom).
    Rationale: the hot path is all NumPy/OpenCV; 90s is extremely conservative
    and only fails if a blocking call re-enters the per-second loop.
    """
    from aese.pipeline import run as aese_run

    config = AESEConfig()
    start = time.perf_counter()
    events = list(aese_run(_fight_clip_stream(), config))
    elapsed = time.perf_counter() - start

    assert elapsed < 90.0, (
        f"Full pipeline took {elapsed:.1f}s for an 81s clip -- budget is 90s. "
        f"A slow call (e.g. a generative VLM) has re-entered the hot path. "
        f"Events produced: {len(events)}"
    )

    # Sanity: pipeline must still produce events (not fail silently)
    assert len(events) >= 1, "Pipeline produced no events -- check for silent failure"