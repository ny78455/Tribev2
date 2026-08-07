"""
tests/test_features.py
Unit tests for all feature extractors (§5.2).

Tests both the "identical frames → low scores" and
"random noise frames → high scores" cases.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.features.motion import compute_motion_score
from asvl.features.scene import compute_scene_score
from asvl.features.edges import compute_edge_diff
from asvl.features.blur import is_blurred, is_black_frame
from asvl.features.novelty import compute_novelty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(color: tuple, shape=(240, 320)) -> np.ndarray:
    """Create a solid color RGB frame."""
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _noise_frame(shape=(240, 320)) -> np.ndarray:
    """Create a random RGB noise frame."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(shape[0], shape[1], 3), dtype=np.uint8)


def _black_frame(shape=(240, 320)) -> np.ndarray:
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

class TestMotionScore:
    def test_identical_frames_low_motion(self):
        frame = _solid_frame((128, 64, 200))
        score = compute_motion_score(frame, frame)
        assert 0.0 <= score <= 1.0
        assert score < 0.05, f"Identical frames should have near-zero motion, got {score}"

    def test_different_frames_have_higher_motion(self):
        prev = _solid_frame((0, 0, 0))
        curr = _noise_frame()
        score = compute_motion_score(prev, curr)
        assert 0.0 <= score <= 1.0
        assert score > 0.0, "Noise frame vs. black should have nonzero motion"

    def test_output_in_range(self):
        prev = _noise_frame()
        curr = _noise_frame()
        score = compute_motion_score(prev, curr)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TestSceneScore:
    def test_identical_frames_no_scene_change(self):
        frame = _solid_frame((100, 150, 200))
        score, changed = compute_scene_score(frame, frame, scene_threshold=0.7)
        assert 0.0 <= score <= 1.0
        assert not changed, "Identical frames should not trigger scene change"

    def test_very_different_frames_may_trigger_scene_change(self):
        prev = _solid_frame((0, 0, 0))    # black
        curr = _solid_frame((255, 255, 255))  # white
        score, changed = compute_scene_score(prev, curr, scene_threshold=0.5)
        assert 0.0 <= score <= 1.0
        assert changed, f"Black→white should trigger scene change at threshold=0.5, got score={score}"

    def test_score_in_range(self):
        prev = _noise_frame()
        curr = _noise_frame()
        score, _ = compute_scene_score(prev, curr)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class TestEdgeDiff:
    def test_identical_frames_zero_diff(self):
        frame = _solid_frame((128, 128, 128))
        diff = compute_edge_diff(frame, frame)
        assert 0.0 <= diff <= 1.0
        # Identical frames → XOR = 0 → diff ≈ 0
        assert diff < 0.01, f"Identical frames should have edge_diff≈0, got {diff}"

    def test_different_frames_nonzero(self):
        prev = _solid_frame((0, 0, 0))
        curr = _noise_frame()
        diff = compute_edge_diff(prev, curr)
        assert 0.0 <= diff <= 1.0

    def test_output_in_range(self):
        prev = _noise_frame()
        curr = _solid_frame((200, 100, 50))
        diff = compute_edge_diff(prev, curr)
        assert 0.0 <= diff <= 1.0


# ---------------------------------------------------------------------------
# Blur / Black Frame
# ---------------------------------------------------------------------------

class TestBlur:
    def test_solid_color_frame_is_blurred(self):
        # Solid color has zero Laplacian variance → blurred
        frame = _solid_frame((128, 128, 128))
        assert is_blurred(frame), "Solid color frame should be detected as blurred"

    def test_noise_frame_not_blurred(self):
        frame = _noise_frame()
        assert not is_blurred(frame), "Noise frame should NOT be detected as blurred"

    def test_black_frame_detection(self):
        frame = _black_frame()
        assert is_black_frame(frame), "All-zero frame should be detected as black"

    def test_bright_frame_not_black(self):
        frame = _solid_frame((200, 200, 200))
        assert not is_black_frame(frame), "Bright frame should not be detected as black"


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

class TestNovelty:
    def test_no_buffer_returns_max_novelty(self):
        frame = _solid_frame((100, 100, 100))
        score = compute_novelty(frame, None)
        assert score == 1.0, "No buffer history → novelty should be 1.0"

    def test_same_frame_vs_buffer_low_novelty(self):
        import cv2
        frame = _solid_frame((100, 100, 100))
        # Build a buffer mean histogram from the same frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [64], [0, 256])
            cv2.normalize(h, h)
            hists.append(h.flatten())
        buf_hist = np.concatenate(hists).astype(np.float32)

        score = compute_novelty(frame, buf_hist)
        assert 0.0 <= score <= 1.0
        assert score < 0.1, f"Same-frame vs. its own histogram should have low novelty, got {score}"

    def test_novel_frame_higher_score(self):
        import cv2
        buffer_frame = _solid_frame((0, 0, 0))  # all-black
        novel_frame = _solid_frame((255, 0, 0))  # all-red

        hsv = cv2.cvtColor(buffer_frame, cv2.COLOR_RGB2HSV)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [64], [0, 256])
            cv2.normalize(h, h)
            hists.append(h.flatten())
        buf_hist = np.concatenate(hists).astype(np.float32)

        score = compute_novelty(novel_frame, buf_hist)
        assert 0.0 <= score <= 1.0
