"""
tests/test_merge_split.py
Tests for EventMerge and EventSplit (§5.12, §5.13).
"""
import numpy as np
import pytest

from aese.types import AESEConfig, Event
from aese.event_merge import should_merge, merge_events, OnlineMerger
from aese.event_split import should_split, split_event
from aese.adapters.embedding import EMBEDDING_DIM


def _make_event(
    event_id: int,
    start_ms: float,
    end_ms: float,
    embedding: np.ndarray = None,
    location: str = "indoor",
    characters: list = None,
) -> Event:
    if embedding is None:
        embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    if characters is None:
        characters = [1]
    return Event(
        event_id=event_id,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        duration_ms=end_ms - start_ms,
        event_embedding=embedding,
        importance=0.5,
        confidence=0.8,
        summary="test event",
        boundary_reason="scene",
        event_type="Scene",
        characters=characters,
        location_label=location,
    )


# ---------------------------------------------------------------------------
# should_merge
# ---------------------------------------------------------------------------
def test_should_merge_identical_scenes():
    """Two events with same location, chars, and very similar embeddings should merge."""
    emb = np.ones(EMBEDDING_DIM, dtype=np.float32) / EMBEDDING_DIM ** 0.5
    a = _make_event(0, 0, 10000, embedding=emb.copy(), location="indoor", characters=[1])
    b = _make_event(1, 10000, 20000, embedding=emb.copy(), location="indoor", characters=[1])
    assert should_merge(a, b, merge_threshold=0.80)


def test_should_not_merge_different_location():
    emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    a = _make_event(0, 0, 10000, embedding=emb.copy(), location="indoor")
    b = _make_event(1, 10000, 20000, embedding=emb.copy(), location="outdoor")
    assert not should_merge(a, b, merge_threshold=0.80)


def test_should_not_merge_different_chars():
    emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    a = _make_event(0, 0, 10000, embedding=emb.copy(), characters=[1])
    b = _make_event(1, 10000, 20000, embedding=emb.copy(), characters=[2])
    assert not should_merge(a, b, merge_threshold=0.80)


def test_should_not_merge_different_embeddings():
    """Events with very different embeddings should not merge even with same location/chars."""
    emb_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    emb_b[-1] = 1.0
    a = _make_event(0, 0, 10000, embedding=emb_a, location="indoor", characters=[1])
    b = _make_event(1, 10000, 20000, embedding=emb_b, location="indoor", characters=[1])
    assert not should_merge(a, b, merge_threshold=0.80)


# ---------------------------------------------------------------------------
# merge_events
# ---------------------------------------------------------------------------
def test_merge_events_preserves_timing():
    emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    a = _make_event(0, 0, 10000, embedding=emb.copy())
    b = _make_event(1, 10000, 20000, embedding=emb.copy())
    merged = merge_events(a, b)
    assert merged.start_time_ms == 0.0
    assert merged.end_time_ms == 20000.0
    assert merged.duration_ms == 20000.0
    assert merged.event_id == 0  # preserves earlier ID


def test_merge_events_embedding_shape():
    emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    a = _make_event(0, 0, 5000, embedding=emb.copy())
    b = _make_event(1, 5000, 10000, embedding=emb.copy())
    merged = merge_events(a, b)
    assert merged.event_embedding.shape == (EMBEDDING_DIM,)


# ---------------------------------------------------------------------------
# OnlineMerger
# ---------------------------------------------------------------------------
def test_online_merger_no_merge():
    """Events with different locations should not be merged."""
    cfg = AESEConfig()
    merger = OnlineMerger(cfg)

    emb_a = np.zeros(EMBEDDING_DIM, dtype=np.float32); emb_a[0] = 1.0
    emb_b = np.zeros(EMBEDDING_DIM, dtype=np.float32); emb_b[-1] = 1.0

    a = _make_event(0, 0, 10000, embedding=emb_a, location="indoor")
    b = _make_event(1, 10000, 20000, embedding=emb_b, location="outdoor")

    result1 = merger.process(a)
    assert result1 is None  # first event held

    result2 = merger.process(b)
    assert result2 is not None  # a finalized
    assert result2.event_id == 0

    final = merger.finalize()
    assert final is not None
    assert final.event_id == 1


# ---------------------------------------------------------------------------
# should_split / split_event
# ---------------------------------------------------------------------------
def test_should_split_over_max():
    event = _make_event(0, 0, 310_000)  # 310s > 300s max
    assert should_split(event, max_duration_s=300.0)


def test_should_not_split_under_max():
    event = _make_event(0, 0, 60_000)  # 60s < 300s
    assert not should_split(event, max_duration_s=300.0)


def test_split_event_produces_two():
    from aese.types import TemporalFeature

    emb = np.ones(EMBEDDING_DIM, dtype=np.float32)
    event = _make_event(0, 0, 310_000, embedding=emb.copy())

    features = []
    for i in range(310):
        features.append(TemporalFeature(
            timestamp_ms=float(i * 1000),
            scene_label="indoor", character_count=1, action_label="static",
            dialogue_present=False, dialogue_text=None, camera_cue=None,
            music_mood="calm", multimodal_embedding=emb.copy(),
            motion_score=0.1, novelty_score=0.2, audio_energy=0.05, spectral_flux=0.0,
        ))

    fused_scores = [(float(i * 1000), 0.1 + (0.5 if i == 150 else 0.0)) for i in range(310)]
    cfg = AESEConfig()
    result = split_event(event, features, fused_scores, cfg, next_event_id=1)
    assert len(result) == 2
    assert result[0].start_time_ms == 0.0
    assert result[1].end_time_ms == 310_000.0
    assert result[0].end_time_ms == result[1].start_time_ms
