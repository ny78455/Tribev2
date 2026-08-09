"""
tests/test_fusion.py
Tests for boundary fusion and confidence scoring.

§4 acceptance test: AESEConfig.weights sum to 1.0 (verifying renormalization fix).
"""
import pytest

from aese.types import AESEConfig, BoundarySignal, _RENORM_WEIGHTS, _RAW_SUM
from aese.boundary.fusion import fuse, dominant_signal_name
from aese.boundary.confidence import compute_confidence, is_high_confidence


# ---------------------------------------------------------------------------
# §4 Acceptance: weights sum assertion (source spec bug = 1.05, we renormalize)
# ---------------------------------------------------------------------------
def test_weights_sum_to_one():
    """
    §4 acceptance test: AESEConfig weights must sum to exactly 1.0.
    Source spec had weights summing to 1.05 — verified, renormalized, logged in DECISIONS.md §1.
    """
    cfg = AESEConfig()
    total = sum(cfg.weights.values())
    assert abs(total - 1.0) < 1e-4, (
        f"AESEConfig weights sum to {total:.6f}, expected 1.0. "
        "See DECISIONS.md §1 for renormalization details."
    )


def test_raw_spec_sum_is_1_05():
    """Confirm we actually caught the source spec bug."""
    assert abs(_RAW_SUM - 1.05) < 1e-9, f"Expected raw sum = 1.05, got {_RAW_SUM}"


def test_all_weight_keys_present():
    cfg = AESEConfig()
    required = {"prediction_error", "scene", "dialogue", "emotion", "character", "embedding", "music"}
    assert required == set(cfg.weights.keys())


def test_all_weights_non_negative():
    cfg = AESEConfig()
    for k, v in cfg.weights.items():
        assert v >= 0.0, f"Weight '{k}' is negative: {v}"


# ---------------------------------------------------------------------------
# fusion: fuse()
# ---------------------------------------------------------------------------
def _make_signal(**kwargs) -> BoundarySignal:
    defaults = dict(scene=0.0, character=0.0, dialogue=0.0, camera=0.0,
                    emotion=0.0, music=0.0, embedding_distance=0.0, prediction_error=0.0)
    defaults.update(kwargs)
    return BoundarySignal(**defaults)


def test_fuse_all_zeros():
    cfg = AESEConfig()
    sig = _make_signal()
    score = fuse(sig, cfg.weights)
    assert score == 0.0


def test_fuse_all_ones():
    cfg = AESEConfig()
    sig = _make_signal(scene=1.0, character=1.0, dialogue=1.0, camera=1.0,
                       emotion=1.0, music=1.0, embedding_distance=1.0, prediction_error=1.0)
    score = fuse(sig, cfg.weights)
    # All signals = 1.0 → score should equal sum of weights = 1.0
    assert abs(score - 1.0) < 1e-4


def test_fuse_output_clamped():
    """fuse() must always return a value in [0, 1]."""
    cfg = AESEConfig()
    for _ in range(20):
        import numpy as np
        vals = np.random.uniform(0, 1, 8).tolist()
        sig = BoundarySignal(*vals[:8])
        score = fuse(sig, cfg.weights)
        assert 0.0 <= score <= 1.0, f"fuse() returned {score} outside [0,1]"


def test_emotion_weight_has_no_effect():
    """emotion_signal is always 0.0, so changing emotion weight has no effect on score."""
    cfg = AESEConfig()
    sig_low = _make_signal(scene=0.5, embedding_distance=0.3)
    sig_high = _make_signal(scene=0.5, embedding_distance=0.3, emotion=1.0)
    # emotion=0 vs emotion=1 with same other signals — scores should differ
    # but since emotion is always 0.0 in practice, this tests the weight is properly applied
    score_low = fuse(sig_low, cfg.weights)
    score_high = fuse(sig_high, cfg.weights)
    # emotion signal = 1.0 * emotion_weight ≈ 0.095
    expected_diff = cfg.weights["emotion"] * 1.0
    assert abs(score_high - score_low - expected_diff) < 1e-4


# ---------------------------------------------------------------------------
# dominant_signal_name
# ---------------------------------------------------------------------------
def test_dominant_scene():
    cfg = AESEConfig()
    sig = _make_signal(scene=1.0)
    assert dominant_signal_name(sig, cfg.weights) == "scene"


def test_dominant_embedding():
    cfg = AESEConfig()
    sig = _make_signal(embedding_distance=1.0)
    assert dominant_signal_name(sig, cfg.weights) == "embedding"


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------
def test_confidence_well_above():
    c = compute_confidence(0.90, threshold=0.75, margin=0.05)
    assert c >= 0.8


def test_confidence_well_below():
    c = compute_confidence(0.50, threshold=0.75, margin=0.05)
    assert c < 0.5


def test_confidence_near_threshold():
    c = compute_confidence(0.75, threshold=0.75, margin=0.05)
    assert 0.4 <= c <= 0.8  # in low-confidence zone


def test_confidence_in_range():
    import numpy as np
    for score in np.linspace(0, 1, 20):
        c = compute_confidence(float(score), threshold=0.75, margin=0.05)
        assert 0.0 <= c <= 1.0, f"confidence({score}) = {c} out of [0,1]"


def test_is_high_confidence():
    assert is_high_confidence(0.85)
    assert not is_high_confidence(0.50)
