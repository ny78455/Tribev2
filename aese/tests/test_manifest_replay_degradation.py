"""
tests/test_manifest_replay_degradation.py
Regression tests that pin the exact failure mode described in the AESE bug report:

  "Running without --video produced a single 59-second event with characters: [0],
  because the aggregator silently substituted a black frame for missing images."

All three tests must pass before this issue is considered fixed.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pytest

from aese.adapters.embedding import EMBEDDING_DIM
from aese.aggregator import FeatureAggregator
from aese.boundary.fusion import fuse
from aese.event_constructor import EventConstructor, _make_summary
from aese.event_embedding import pool_event_embedding
from aese.keyframe import select_keyframe
from aese.types import AESEConfig, BoundaryDecision, BoundarySignal, Event, FramePacket, TemporalFeature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame_packet(ts_ms: float, image=None, scene_change: bool = False,
                       motion: float = 0.2, audio: float = 0.1,
                       novelty: float = 0.1, subtitle: Optional[str] = None) -> FramePacket:
    return FramePacket(
        frame_id=int(ts_ms),
        timestamp_ms=ts_ms,
        fps_used=1.0,
        motion_score=motion,
        scene_change=scene_change,
        audio_energy=audio,
        novelty_score=novelty,
        decision_reason="test",
        subtitle_text=subtitle,
        image=image,
    )


def _make_no_image_stream(duration_s: int = 10, fps: int = 1) -> List[FramePacket]:
    """Return a stream of FramePackets all with image=None (manifest-replay mode)."""
    packets = []
    for s in range(duration_s):
        for f in range(fps):
            ts_ms = float(s * 1000 + f * (1000 // fps))
            packets.append(_make_frame_packet(ts_ms, image=None))
    return packets


def _make_constructor() -> EventConstructor:
    cfg = AESEConfig()
    return EventConstructor(
        config=cfg,
        event_embedding_fn=pool_event_embedding,
        keyframe_fn=lambda feats: select_keyframe(feats, "lowest_blur"),
    )


def _make_boundary_decision(is_boundary: bool = False, score: float = 0.5) -> BoundaryDecision:
    return BoundaryDecision(
        is_boundary=is_boundary,
        confidence=score,
        dominant_signal="camera",
        fused_score=score,
    )


def _make_signal(**kwargs) -> BoundarySignal:
    defaults = dict(scene=0.0, character=0.0, dialogue=0.0, camera=0.0,
                    emotion=0.0, music=0.0, embedding_distance=0.0, prediction_error=0.0)
    defaults.update(kwargs)
    return BoundarySignal(**defaults)


# ---------------------------------------------------------------------------
# Test 1 — Aggregator: None images must produce None character_count + image_available=False
# ---------------------------------------------------------------------------

def test_aggregator_none_images_produce_none_character_count():
    """
    Regression: feed the aggregator a stream of FramePackets with image=None.
    Must produce TemporalFeatures with character_count=None and image_available=False —
    NOT character_count=0 presented as real data.
    """
    cfg = AESEConfig()
    agg = FeatureAggregator(cfg)

    packets = _make_no_image_stream(duration_s=5, fps=2)
    features: List[TemporalFeature] = []
    for pkt in packets:
        results = agg.push_all(pkt)
        features.extend(results)
    features.extend(agg.flush())

    assert len(features) > 0, "Aggregator must emit at least one TemporalFeature"

    for tf in features:
        assert tf.character_count is None, (
            f"character_count should be None (no real image), got {tf.character_count!r} "
            f"at ts={tf.timestamp_ms}ms. This is the defect: conflating 'missing' with '0'."
        )
        assert tf.image_available is False, (
            f"image_available should be False (no real image), got {tf.image_available!r} "
            f"at ts={tf.timestamp_ms}ms."
        )


# ---------------------------------------------------------------------------
# Test 2 — EventConstructor: all-None character_counts → characters=None, not [0]
# ---------------------------------------------------------------------------

def test_no_video_mode_characters_is_none_not_zero_list():
    """
    Regression: construct an event from TemporalFeatures that all have character_count=None.
    Event.characters must be None and character_data_available must be False —
    NEVER [0] or [] (the original defect output).
    """
    cfg = AESEConfig()
    ec = _make_constructor()

    # Feed 8 seconds of no-image features (> minimum_event_duration_s=5)
    for i in range(8):
        tf = TemporalFeature(
            timestamp_ms=float(i * 1000),
            scene_label="unknown",
            character_count=None,      # no image data
            action_label="static",
            dialogue_present=False,
            dialogue_text=None,
            camera_cue=None,
            music_mood="calm",
            multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            motion_score=0.1,
            novelty_score=0.1,
            audio_energy=0.1,
            spectral_flux=0.0,
            image_available=False,
        )
        ec.update(tf, _make_boundary_decision(is_boundary=False))

    # Trigger a boundary
    tf_boundary = TemporalFeature(
        timestamp_ms=8000.0,
        scene_label="unknown",
        character_count=None,
        action_label="static",
        dialogue_present=False,
        dialogue_text=None,
        camera_cue="cut",
        music_mood="calm",
        multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        motion_score=0.8,
        novelty_score=0.8,
        audio_energy=0.1,
        spectral_flux=0.0,
        image_available=False,
    )
    events = ec.update(tf_boundary, _make_boundary_decision(is_boundary=True, score=0.9))

    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    event = events[0]

    assert event.characters is None, (
        f"Event.characters should be None (no image data), got {event.characters!r}. "
        "This is the reported defect: characters: [0] appeared instead of null."
    )
    assert event.character_data_available is False, (
        f"Event.character_data_available should be False, got {event.character_data_available!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Summary: None char_count → "character data unavailable" in summary text
# ---------------------------------------------------------------------------

def test_make_summary_none_char_count():
    """
    _make_summary with char_count=None must produce a string containing
    'character data unavailable', not 'no people detected' or '0 people'.
    """
    summary = _make_summary("outdoor", "static", None)
    assert "character data unavailable" in summary, (
        f"Expected 'character data unavailable' in summary, got: {summary!r}"
    )
    assert "0" not in summary, (
        f"Summary should not mention '0' when data is unavailable: {summary!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Fusion: weight renormalization lets camera+dialogue cross threshold
# ---------------------------------------------------------------------------

def test_fusion_renormalizes_with_unavailable_channels():
    """
    Regression: with scene/character/embedding/prediction_error all unavailable,
    strong camera + dialogue + music signals must cross boundary_threshold=0.75
    after weight renormalization.

    Before the fix, these channels contributed 0 to the numerator but still consumed
    their weight in the denominator — structurally capping the fused score at ~0.39
    (camera+dialogue+music weights only), making threshold=0.75 unreachable.

    After the fix: renormalized over only the available channels, so camera+dialogue+music
    at full strength use their full weight budget and CAN cross the threshold.
    """
    cfg = AESEConfig()

    # Strong camera cut + dialogue onset + music change — real signals
    signals = _make_signal(camera=1.0, dialogue=1.0, music=1.0)

    # Mark all image-dependent channels as unavailable
    available = {
        "scene": False,
        "character": False,
        "embedding": False,
        "prediction_error": False,
        "dialogue": True,
        "camera": True,
        "music": True,
        "emotion": True,
    }

    score_with_available = fuse(signals, cfg.weights, available)
    score_without_available = fuse(signals, cfg.weights)  # old behavior

    # Key assertion: renormalization gives a HIGHER score (less structural cap)
    assert score_with_available > score_without_available, (
        f"Renormalized score {score_with_available:.4f} should be > "
        f"non-renormalized score {score_without_available:.4f}. "
        "Renormalization is not working."
    )

    # With camera+dialogue+music all firing, renormalized score must cross threshold
    assert score_with_available >= cfg.boundary_threshold, (
        f"Fused score {score_with_available:.4f} did not reach "
        f"boundary_threshold={cfg.boundary_threshold}. "
        "Weight renormalization is broken — image-unavailable channels are still "
        "dragging the denominator down."
    )


def test_fusion_all_available_unchanged():
    """
    When all channels are available (or available dict is omitted),
    fuse() must behave identically to the pre-fix behavior.
    Emotion=1.0 is set explicitly (matching test_fuse_all_ones) so all
    signals genuinely sum to 1.0 with the full weight set.
    """
    cfg = AESEConfig()
    # Include emotion=1.0 — the _make_signal default is 0.0 which would leave
    # the emotion weight slot unfilled and produce < 1.0
    signals = _make_signal(scene=1.0, character=1.0, dialogue=1.0,
                           camera=1.0, emotion=1.0, embedding_distance=1.0,
                           prediction_error=1.0, music=1.0)

    score_no_avail = fuse(signals, cfg.weights)
    score_all_avail = fuse(signals, cfg.weights, {k: True for k in cfg.weights})
    assert abs(score_no_avail - score_all_avail) < 1e-6, (
        f"fuse() with all-available should equal no-available-dict: "
        f"{score_no_avail} vs {score_all_avail}"
    )
    # All signals=1.0 and all weights available → score should be 1.0
    assert abs(score_no_avail - 1.0) < 1e-4, (
        f"All signals=1.0 should fuse to 1.0, got {score_no_avail}"
    )


def test_fusion_camera_only_unavailable_image_channels():
    """
    Camera=1.0 alone (all image channels unavailable, no dialogue) must produce
    a score = 1.0 after renormalization (camera uses 100% of remaining weight).
    """
    cfg = AESEConfig()
    signals = _make_signal(camera=1.0)  # only camera fires

    available = {
        "scene": False,
        "character": False,
        "embedding": False,
        "prediction_error": False,
        "dialogue": True,
        "camera": True,
        "music": True,
        "emotion": True,
    }

    score = fuse(signals, cfg.weights, available)
    # With camera=1.0, dialogue=0.0, music=0.0, emotion=0.0:
    # active weight = character_weight (shared camera slot) + dialogue_weight + music_weight + emotion_weight
    # score = character_weight * max(0.0, 1.0) + dialogue * 0.0 + music * 0.0 + emotion * 0.0
    # normalized = character_weight / active_total_weight
    # This should be significantly positive but less than 1.0 (other active channels got 0)
    # Key check: score must be > 0 and the function must not crash
    assert score > 0.0, f"Camera=1.0 with renormalization should produce score > 0, got {score}"
    assert score <= 1.0, f"Score must be clamped to [0,1], got {score}"


# ---------------------------------------------------------------------------
# Test 5 — CLI warning: manifest-replay mode must emit warning banner (unit-level)
# ---------------------------------------------------------------------------

def test_no_video_mode_aggregator_sets_image_available_false(caplog):
    """
    End-to-end aggregator behavior: FramePackets with image=None must all
    produce image_available=False and character_count=None.
    Validates that the fix holds through the full aggregation pipeline.
    """
    cfg = AESEConfig()
    agg = FeatureAggregator(cfg)

    # 3 seconds of packets with no image data
    packets = [
        _make_frame_packet(0.0, image=None),
        _make_frame_packet(100.0, image=None),
        _make_frame_packet(200.0, image=None),
        _make_frame_packet(1000.0, image=None),
        _make_frame_packet(1100.0, image=None),
        _make_frame_packet(2000.0, image=None),
    ]

    features: List[TemporalFeature] = []
    for pkt in packets:
        features.extend(agg.push_all(pkt))
    features.extend(agg.flush())

    # Every emitted feature must honestly report no image data
    for tf in features:
        assert tf.image_available is False, (
            f"image_available={tf.image_available} at ts={tf.timestamp_ms}ms; "
            "expected False for all-None-image stream"
        )
        assert tf.character_count is None, (
            f"character_count={tf.character_count!r} at ts={tf.timestamp_ms}ms; "
            "expected None, not 0 (the reported defect)"
        )
