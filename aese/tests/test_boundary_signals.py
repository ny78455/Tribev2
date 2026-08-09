"""
tests/test_boundary_signals.py
Acceptance tests for boundary signal functions (§5.3, §5.4).

Key tests:
  - emotion_signal always returns 0.0 (critical honesty check)
  - All signals return float in [0, 1]
  - embedding_distance: identical→0, orthogonal→0.5, opposite→1.0
"""
import numpy as np
import pytest

from aese.boundary.signals import (
    camera_signal,
    character_signal,
    dialogue_signal,
    emotion_signal,
    music_signal,
    scene_signal,
)
from aese.boundary.embedding_change import embedding_distance
from aese.adapters.embedding import EMBEDDING_DIM
from aese.types import TemporalFeature


def _make_feature(**kwargs) -> TemporalFeature:
    defaults = dict(
        timestamp_ms=0.0,
        scene_label="indoor",
        character_count=1,
        action_label="static",
        dialogue_present=False,
        dialogue_text=None,
        camera_cue=None,
        music_mood="calm",
        multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        motion_score=0.0,
        novelty_score=0.0,
        audio_energy=0.0,
        spectral_flux=0.0,
    )
    defaults.update(kwargs)
    return TemporalFeature(**defaults)


# ---------------------------------------------------------------------------
# Critical: emotion_signal must always return 0.0
# ---------------------------------------------------------------------------
def test_emotion_signal_always_zero():
    """
    §5.3 critical honesty check: emotion_signal must return 0.0.
    No emotion model is in scope — any non-zero value would corrupt downstream fusion.
    """
    curr = _make_feature(novelty_score=1.0, motion_score=1.0, audio_energy=1.0)
    prev = _make_feature()
    result = emotion_signal(curr, prev)
    assert result == 0.0, (
        f"CRITICAL: emotion_signal returned {result}, expected 0.0. "
        "This is a documented stub (DECISIONS.md §8). "
        "A non-zero value would fabricate signal data with no underlying model."
    )


# ---------------------------------------------------------------------------
# scene_signal
# ---------------------------------------------------------------------------
def test_scene_signal_change():
    curr = _make_feature(scene_label="outdoor")
    prev = _make_feature(scene_label="indoor")
    assert scene_signal(curr, prev) == 1.0


def test_scene_signal_no_change():
    curr = _make_feature(scene_label="indoor")
    prev = _make_feature(scene_label="indoor")
    assert scene_signal(curr, prev) == 0.0


# ---------------------------------------------------------------------------
# character_signal
# ---------------------------------------------------------------------------
def test_character_signal_large_change():
    curr = _make_feature(character_count=2)
    prev = _make_feature(character_count=0)
    result = character_signal(curr, prev)
    assert result == 1.0  # 2/2 = 1.0, capped at 1.0


def test_character_signal_no_change():
    curr = _make_feature(character_count=1)
    prev = _make_feature(character_count=1)
    assert character_signal(curr, prev) == 0.0


def test_character_signal_in_range():
    for curr_n in range(0, 4):
        for prev_n in range(0, 4):
            curr = _make_feature(character_count=curr_n)
            prev = _make_feature(character_count=prev_n)
            val = character_signal(curr, prev)
            assert 0.0 <= val <= 1.0, f"character_signal out of [0,1]: {val}"


# ---------------------------------------------------------------------------
# dialogue_signal
# ---------------------------------------------------------------------------
def test_dialogue_signal_onset():
    curr = _make_feature(dialogue_present=True)
    prev = _make_feature(dialogue_present=False)
    assert dialogue_signal(curr, prev) == 1.0


def test_dialogue_signal_cessation():
    curr = _make_feature(dialogue_present=False)
    prev = _make_feature(dialogue_present=True)
    assert dialogue_signal(curr, prev) == 1.0


def test_dialogue_signal_no_change():
    for val in [True, False]:
        curr = _make_feature(dialogue_present=val)
        prev = _make_feature(dialogue_present=val)
        assert dialogue_signal(curr, prev) == 0.0


# ---------------------------------------------------------------------------
# camera_signal
# ---------------------------------------------------------------------------
def test_camera_signal_cut():
    curr = _make_feature(camera_cue="cut")
    assert camera_signal(curr) == 1.0


def test_camera_signal_fade():
    curr = _make_feature(camera_cue="fade")
    assert camera_signal(curr) == 1.0


def test_camera_signal_black():
    # "black" alone does not trigger camera signal (not a cut/fade)
    curr = _make_feature(camera_cue="black")
    assert camera_signal(curr) == 0.0


def test_camera_signal_none():
    curr = _make_feature(camera_cue=None)
    assert camera_signal(curr) == 0.0


# ---------------------------------------------------------------------------
# music_signal
# ---------------------------------------------------------------------------
def test_music_signal_change():
    curr = _make_feature(music_mood="energetic")
    prev = _make_feature(music_mood="calm")
    assert music_signal(curr, prev) == 1.0


def test_music_signal_no_change():
    curr = _make_feature(music_mood="calm")
    prev = _make_feature(music_mood="calm")
    assert music_signal(curr, prev) == 0.0


# ---------------------------------------------------------------------------
# §5.4 Embedding distance acceptance test
# ---------------------------------------------------------------------------
def test_embedding_distance_identical():
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dist = embedding_distance(emb, emb.copy(), metric="cosine")
    assert abs(dist) < 1e-6, f"Identical embeddings should have distance≈0, got {dist}"


def test_embedding_distance_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    dist = embedding_distance(a, b, metric="cosine")
    # (1 - 0) / 2 = 0.5
    assert abs(dist - 0.5) < 1e-6, f"Orthogonal embeddings should have distance=0.5, got {dist}"


def test_embedding_distance_opposite():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    dist = embedding_distance(a, b, metric="cosine")
    # (1 - (-1)) / 2 = 1.0
    assert abs(dist - 1.0) < 1e-6, f"Opposite embeddings should have distance=1.0, got {dist}"


def test_embedding_distance_in_range():
    """All distances must be in [0, 1]."""
    for _ in range(10):
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        dist = embedding_distance(a, b, metric="cosine")
        assert 0.0 <= dist <= 1.0, f"Distance {dist} out of [0,1]"


def test_embedding_distance_zero_vector():
    """Zero vectors should not crash — return 0."""
    z = np.zeros(64, dtype=np.float32)
    b = np.random.randn(64).astype(np.float32)
    dist = embedding_distance(z, b, metric="cosine")
    assert dist == 0.0
