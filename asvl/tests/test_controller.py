"""
tests/test_controller.py
Acceptance tests for asvl.controller (§5.6).

Verifies exact FPS tier mappings per the contract table.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.controller import compute_importance, importance_to_fps, get_decision_reason
from asvl.types import ASVLConfig

_DEFAULT_WEIGHTS = ASVLConfig().weights


class TestComputeImportance:
    def test_all_zero_inputs(self):
        score = compute_importance(0.0, False, 0.0, 0.0, 0.0, _DEFAULT_WEIGHTS)
        assert score == 0.0

    def test_all_max_inputs(self):
        score = compute_importance(1.0, True, 1.0, 1.0, 1.0, _DEFAULT_WEIGHTS)
        assert abs(score - 1.0) < 1e-6

    def test_formula_correctness(self):
        w = _DEFAULT_WEIGHTS
        # Expected: 0.3*0.5 + 0.25*1 + 0.2*0.8 + 0.1*0.3 + 0.15*0.6
        expected = 0.3 * 0.5 + 0.25 * 1.0 + 0.2 * 0.8 + 0.1 * 0.3 + 0.15 * 0.6
        result = compute_importance(0.5, True, 0.8, 0.3, 0.6, w)
        assert abs(result - expected) < 1e-6

    def test_output_clipped_to_zero_one(self):
        # With extreme weights this could theoretically overflow, must be clipped
        extreme_weights = {"motion": 1.0, "scene": 1.0, "audio": 1.0, "subtitle": 1.0, "novelty": 1.0}
        score = compute_importance(1.0, True, 1.0, 1.0, 1.0, extreme_weights)
        assert 0.0 <= score <= 1.0


class TestImportanceToFps:
    """Exact FPS tier table from §5.6."""

    def test_tier_0_5(self):
        assert importance_to_fps(0.0) == 0.5
        assert importance_to_fps(0.1) == 0.5
        assert importance_to_fps(0.19) == 0.5

    def test_tier_1_0(self):
        assert importance_to_fps(0.2) == 1.0
        assert importance_to_fps(0.3) == 1.0
        assert importance_to_fps(0.39) == 1.0

    def test_tier_2_0(self):
        assert importance_to_fps(0.4) == 2.0
        assert importance_to_fps(0.5) == 2.0
        assert importance_to_fps(0.59) == 2.0

    def test_tier_5_0(self):
        assert importance_to_fps(0.6) == 5.0
        assert importance_to_fps(0.7) == 5.0
        assert importance_to_fps(0.79) == 5.0

    def test_tier_10_0(self):
        assert importance_to_fps(0.8) == 10.0
        assert importance_to_fps(0.9) == 10.0
        assert importance_to_fps(1.0) == 10.0


class TestDecisionReason:
    def test_static_reason_for_low_importance(self):
        reason = get_decision_reason(0.0, False, 0.0, 0.0, 0.0, _DEFAULT_WEIGHTS, 0.0)
        assert reason == "Static / dialogue"

    def test_motion_dominant_reason(self):
        reason = get_decision_reason(1.0, False, 0.0, 0.0, 0.0, _DEFAULT_WEIGHTS, 0.8)
        assert reason == "Fast motion"

    def test_scene_dominant_reason(self):
        reason = get_decision_reason(0.0, True, 0.0, 0.0, 0.0, _DEFAULT_WEIGHTS, 0.5)
        assert reason == "Scene transition"

    def test_audio_dominant_reason(self):
        # Audio weight=0.2, with high energy and zero others
        reason = get_decision_reason(0.0, False, 1.0, 0.0, 0.0, _DEFAULT_WEIGHTS, 0.3)
        assert reason == "High audio energy"

    def test_novelty_dominant_reason(self):
        reason = get_decision_reason(0.0, False, 0.0, 0.0, 1.0, _DEFAULT_WEIGHTS, 0.3)
        assert reason == "Novel content"
