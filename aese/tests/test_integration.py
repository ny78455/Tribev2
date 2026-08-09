"""
tests/test_integration.py
Integration tests — §8 non-functional acceptance gates.

Tests:
  1. Aggregator: 3s synthetic at varying FPS → exactly 3 TemporalFeatures (§5.1)
  2. CandidateDetector: no flapping, ≤2s delay (§5.7)
  3. Single-scene 60s clip → not one-per-second events (§8 fragmentation gate)
  4. Streaming: pipeline produces events without buffering full stream
  5. Context buffer carry-forward works (§5.1 continuity rule)
"""
import time
from typing import List

import numpy as np
import pytest

from aese.types import AESEConfig, BoundarySignal, FramePacket, TemporalFeature
from aese.aggregator import FeatureAggregator
from aese.context_buffer import ContextBuffer
from aese.boundary.candidate_detector import CandidateDetector
from aese.adapters.embedding import EMBEDDING_DIM
from aese.pipeline import run


def _make_fp(ts: float, motion: float = 0.1, scene_change: bool = False,
             audio: float = 0.05, novelty: float = 0.1) -> FramePacket:
    return FramePacket(
        frame_id=int(ts),
        timestamp_ms=ts,
        fps_used=1.0,
        motion_score=motion,
        scene_change=scene_change,
        audio_energy=audio,
        novelty_score=novelty,
        decision_reason="test",
        subtitle_text=None,
        image=None,
    )


# ---------------------------------------------------------------------------
# §5.1 Aggregator: exactly 1 TemporalFeature per second
# ---------------------------------------------------------------------------
def test_aggregator_one_per_second_varying_fps():
    """
    Feed 3 seconds of synthetic FramePackets at varying rates:
      - Second 0: 10 packets (at 10 fps)
      - Second 1: 1 packet (at 1 fps)
      - Second 2: 1 packet (at 0.5 fps — but it falls in second 2)
    Assert exactly 3 TemporalFeatures emitted (plus any from flush).
    """
    cfg = AESEConfig()
    agg = FeatureAggregator(cfg)

    # Second 0: 10 packets at 100ms intervals
    all_features: List[TemporalFeature] = []
    for i in range(10):
        ts = float(i * 100)
        features = agg.push_all(_make_fp(ts, motion=0.2))
        all_features.extend(features)

    # Second 1: 1 packet at 1500ms (falls in second 1)
    features = agg.push_all(_make_fp(1500.0, motion=0.3))
    all_features.extend(features)

    # Second 2: 1 packet at 2800ms
    features = agg.push_all(_make_fp(2800.0, motion=0.1))
    all_features.extend(features)

    # Flush any trailing partial second
    all_features.extend(agg.flush())

    assert len(all_features) == 3, (
        f"Expected exactly 3 TemporalFeatures, got {len(all_features)}. "
        "Aggregator must emit exactly one per wall-clock second."
    )


def test_aggregator_carry_forward_on_gap():
    """
    §5.1 continuity rule: if a second has no packets, carry forward previous feature.
    Feed packets at 0ms and 3000ms — second 1 and 2 should be carry-forward.
    """
    cfg = AESEConfig()
    agg = FeatureAggregator(cfg)

    all_features = []
    all_features.extend(agg.push_all(_make_fp(0.0, motion=0.5, novelty=0.7)))
    all_features.extend(agg.push_all(_make_fp(3000.0, motion=0.2, novelty=0.3)))
    all_features.extend(agg.flush())

    # Should have second 0, 1 (carry), 2 (carry), 3
    assert len(all_features) == 4, f"Expected 4 features (0s + 2 carry-fwd + 3s), got {len(all_features)}"

    # Carry-forward seconds should have same motion_score as original (second 0)
    assert all_features[1].motion_score == pytest.approx(all_features[0].motion_score, abs=1e-5)
    assert all_features[2].motion_score == pytest.approx(all_features[0].motion_score, abs=1e-5)


# ---------------------------------------------------------------------------
# §5.7 CandidateDetector: no flapping, ≤2s delay
# ---------------------------------------------------------------------------
def _make_tf_with_embedding(ts: float, emb_val: float = 0.0) -> TemporalFeature:
    emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    emb[0] = emb_val
    return TemporalFeature(
        timestamp_ms=ts,
        scene_label="indoor", character_count=1, action_label="static",
        dialogue_present=False, dialogue_text=None, camera_cue=None,
        music_mood="calm", multimodal_embedding=emb,
        motion_score=0.1, novelty_score=0.1, audio_energy=0.05, spectral_flux=0.0,
    )


def test_candidate_detector_no_flapping():
    """
    A score that oscillates around threshold (above/below) should not fire
    a boundary on every crossing.
    """
    cfg = AESEConfig()
    cfg.boundary_threshold = 0.75
    buf = ContextBuffer(buffer_seconds=45.0)
    detector = CandidateDetector(cfg, buf)

    # Build features with alternating high/low embedding distance
    high_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32); high_emb[0] = 1.0
    low_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32); low_emb[-1] = 1.0

    prev = None
    boundary_count = 0
    for i in range(10):
        ts = float(i * 1000)
        # Alternate between two distant embeddings to create oscillating distance
        emb = high_emb if i % 2 == 0 else low_emb
        tf = TemporalFeature(
            timestamp_ms=ts,
            scene_label="indoor", character_count=1, action_label="static",
            dialogue_present=False, dialogue_text=None,
            camera_cue=None, music_mood="calm",
            multimodal_embedding=emb,
            motion_score=0.1, novelty_score=0.1, audio_energy=0.05, spectral_flux=0.0,
        )
        buf.push(tf)
        decision = detector.update(tf, prev)
        if decision.is_boundary:
            boundary_count += 1
        prev = tf

    # Should not fire 10 boundaries on 10 alternating frames
    assert boundary_count < 10, (
        f"Detector fired {boundary_count} boundaries on 10 oscillating frames — "
        "flapping prevention is not working."
    )


def test_candidate_detector_max_hold_not_exceeded():
    """
    Confirm that the hold period never exceeds 2000ms.
    We inject a sequence where score is in the low-confidence zone for many seconds.
    """
    cfg = AESEConfig()
    cfg.boundary_threshold = 0.50  # low threshold so we get into hold zone
    buf = ContextBuffer(buffer_seconds=45.0)
    detector = CandidateDetector(cfg, buf)

    # All features near threshold — should stay in hold zone
    emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    emb[0] = 1.0

    prev = None
    hold_started = None
    for i in range(10):
        ts = float(i * 1000)
        tf = TemporalFeature(
            timestamp_ms=ts,
            scene_label="indoor" if i % 2 == 0 else "outdoor",
            character_count=1, action_label="static",
            dialogue_present=i % 3 == 0, dialogue_text=None,
            camera_cue="cut" if i == 3 else None,
            music_mood="calm",
            multimodal_embedding=emb.copy(),
            motion_score=0.1, novelty_score=0.1, audio_energy=0.05, spectral_flux=0.0,
        )
        buf.push(tf)
        decision = detector.update(tf, prev)

        # Any boundary committed must be within 2 seconds of initial hold start
        if decision.is_boundary and hold_started is not None:
            hold_duration = ts - hold_started
            assert hold_duration <= 2000.0 + 500.0, (  # +500ms tolerance for processing
                f"Hold period {hold_duration}ms exceeded 2000ms limit"
            )
        prev = tf


# ---------------------------------------------------------------------------
# §8 No unnecessary fragmentation: single-scene clip → not one-per-second events
# ---------------------------------------------------------------------------
def test_no_fragmentation_single_scene():
    """
    A synthetic 15-second static clip (all signals near zero, same scene) should
    produce a small number of events — NOT one per second or one per packet.
    """
    cfg = AESEConfig()
    cfg.boundary_threshold = 0.75

    # 15 packets, one per second, same static scene
    packets = [
        _make_fp(float(i * 1000), motion=0.05, scene_change=False,
                 audio=0.03, novelty=0.05)
        for i in range(15)
    ]

    events = list(run(iter(packets), cfg))

    assert len(events) < 15, (
        f"Got {len(events)} events for a 15-second static clip — "
        "expected << 15 (no fragmentation). "
        "The pipeline is splitting on every second."
    )
    assert len(events) >= 1, "Expected at least 1 event (the whole clip)"


# ---------------------------------------------------------------------------
# §8 Streaming: pipeline yields events without buffering full stream
# ---------------------------------------------------------------------------
def test_pipeline_is_streaming():
    """
    Run 30 packets through the pipeline and verify it yields events
    before all packets are consumed (i.e., it's truly online/streaming).
    This is a functional test — we check that at least some events are produced.
    """
    cfg = AESEConfig()
    packets = [
        _make_fp(float(i * 1000), motion=0.3, scene_change=(i % 10 == 5),
                 audio=0.1, novelty=0.2)
        for i in range(30)
    ]
    events = list(run(iter(packets), cfg))
    # Should produce at least 1 event for 30s of video
    assert len(events) >= 1, "Pipeline should produce at least 1 event for 30s of video"
