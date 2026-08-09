"""
tests/test_event_constructor.py
Tests for EventConstructor, keyframe selection, and event embedding.
"""
import numpy as np
import pytest

from aese.types import AESEConfig, BoundaryDecision, TemporalFeature
from aese.event_constructor import EventConstructor, _make_summary
from aese.event_embedding import pool_event_embedding
from aese.keyframe import select_keyframe
from aese.adapters.embedding import EMBEDDING_DIM


def _make_feature(ts: float, motion: float = 0.1, novelty: float = 0.2, **kwargs) -> TemporalFeature:
    defaults = dict(
        timestamp_ms=ts,
        scene_label="indoor",
        character_count=1,
        action_label="static",
        dialogue_present=True,
        dialogue_text="hello",
        camera_cue=None,
        music_mood="calm",
        multimodal_embedding=np.random.randn(EMBEDDING_DIM).astype(np.float32),
        motion_score=motion,
        novelty_score=novelty,
        audio_energy=0.1,
        spectral_flux=0.0,
    )
    defaults.update(kwargs)
    return TemporalFeature(**defaults)


def _make_decision(is_boundary=False, confidence=0.5, score=0.5) -> BoundaryDecision:
    return BoundaryDecision(
        is_boundary=is_boundary,
        confidence=confidence,
        dominant_signal="scene",
        fused_score=score,
    )


def _make_constructor():
    cfg = AESEConfig()
    return EventConstructor(
        config=cfg,
        event_embedding_fn=pool_event_embedding,
        keyframe_fn=lambda feats: select_keyframe(feats, "lowest_blur"),
    )


# ---------------------------------------------------------------------------
# Event embedding pooling
# ---------------------------------------------------------------------------
def test_pool_event_embedding_shape():
    features = [_make_feature(float(i * 1000)) for i in range(5)]
    emb = pool_event_embedding(features)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (EMBEDDING_DIM,)
    assert emb.dtype == np.float32


def test_pool_event_embedding_empty():
    """Empty feature list should return zero vector, not crash."""
    emb = pool_event_embedding([])
    assert isinstance(emb, np.ndarray)
    assert len(emb) > 0


def test_pool_event_embedding_single():
    feat = _make_feature(0.0)
    emb = pool_event_embedding([feat])
    assert np.allclose(emb, feat.multimodal_embedding, atol=1e-5)


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------
def test_keyframe_lowest_blur():
    features = [
        _make_feature(0.0, motion=0.9),
        _make_feature(1000.0, motion=0.1),   # lowest motion = sharpest
        _make_feature(2000.0, motion=0.5),
    ]
    kf = select_keyframe(features, strategy="lowest_blur")
    # Should return embedding of the lowest-motion feature
    assert np.allclose(kf, features[1].multimodal_embedding, atol=1e-5)


def test_keyframe_center():
    features = [_make_feature(float(i * 1000)) for i in range(5)]
    kf = select_keyframe(features, strategy="center")
    assert np.allclose(kf, features[2].multimodal_embedding, atol=1e-5)


def test_keyframe_highest_novelty():
    features = [
        _make_feature(0.0, novelty=0.1),
        _make_feature(1000.0, novelty=0.9),  # highest novelty
        _make_feature(2000.0, novelty=0.3),
    ]
    kf = select_keyframe(features, strategy="highest_novelty")
    assert np.allclose(kf, features[1].multimodal_embedding, atol=1e-5)


def test_keyframe_empty():
    assert select_keyframe([], strategy="center") is None


def test_keyframe_invalid_strategy():
    features = [_make_feature(0.0)]
    with pytest.raises(ValueError):
        select_keyframe(features, strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# Summary template
# ---------------------------------------------------------------------------
def test_make_summary_format():
    s = _make_summary("indoor", "static", 2)
    assert "indoor" in s
    assert "2" in s


def test_make_summary_zero_chars():
    s = _make_summary("outdoor", "fast_action", 0)
    assert "no people detected" in s


# ---------------------------------------------------------------------------
# EventConstructor
# ---------------------------------------------------------------------------
def test_constructor_no_boundary():
    """Without a boundary, no events should be emitted."""
    ec = _make_constructor()
    results = []
    for i in range(5):
        tf = _make_feature(float(i * 1000))
        events = ec.update(tf, _make_decision(is_boundary=False))
        results.extend(events)
    assert len(results) == 0


def test_constructor_emits_on_boundary():
    """A confirmed boundary should emit an event."""
    ec = _make_constructor()
    # Accumulate 6 seconds (> min_duration_s=5)
    for i in range(6):
        tf = _make_feature(float(i * 1000))
        ec.update(tf, _make_decision(is_boundary=False))
    # Trigger boundary
    tf = _make_feature(6000.0)
    events = ec.update(tf, _make_decision(is_boundary=True, confidence=0.9))
    assert len(events) == 1
    ev = events[0]
    assert ev.start_time_ms == 0.0
    assert ev.end_time_ms == 6000.0
    assert ev.duration_ms == 6000.0
    assert ev.confidence == pytest.approx(0.9)


def test_constructor_close_returns_event():
    """close() should flush the final event."""
    ec = _make_constructor()
    for i in range(6):
        ec.update(_make_feature(float(i * 1000)), _make_decision(is_boundary=False))
    final = ec.close()
    assert final is not None
    assert final.duration_ms > 0
