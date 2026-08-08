"""
tests/test_regression_comedy_clip.py
Regression fixture for ASVL v1.1 — uses comedy.mp4 as a real-world ground-truth clip.

Clip spec:
  - Duration: 60s, 720x1280, 30fps
  - 00:00–00:30: static dialogue segment (expected fps_used <= 1.0)
  - 00:31–01:00: high-action car-roll/sprint/jump (expected fps_used >= 5.0 for ≥1 packet)
  - 4 known hard cuts at 00:26, 00:33, 00:40, 00:49

Definition of done (ASVL v1.1):
  - All five tests below pass.
  - Full-clip runtime < 60 seconds.
  - DECISIONS.md documents every threshold/weight/normalization change.

Run with:
    pytest tests/test_regression_comedy_clip.py -v
"""

import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ground truth constants
# ---------------------------------------------------------------------------
COMEDY_MP4 = Path(__file__).parent.parent.parent / "comedy.mp4"

GROUND_TRUTH = {
    "static_segment_ms": (0, 30_000),          # expect fps_used <= 1.0 for non-scene packets
    "action_segment_ms": (31_000, 60_000),      # expect fps_used >= 5.0 for at least one packet
    "known_cuts_ms": [26_000, 33_000, 40_000, 49_000],  # scene_change=True within ±700ms
    "max_runtime_seconds": 60,                  # full clip must process in under 60s
    "scene_cut_tolerance_ms": 700,              # ±700ms: at 30fps, scheduler granularity ~633ms
}

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _require_clip() -> None:
    """Skip the test if comedy.mp4 is not present."""
    if not COMEDY_MP4.exists():
        pytest.skip(
            f"comedy.mp4 not found at {COMEDY_MP4}. "
            "Place the clip at the project root to run regression tests."
        )


def _run_pipeline():
    """Run the full ASVL pipeline on comedy.mp4 and collect all emitted packets."""
    from asvl.config import load_config
    from asvl.pipeline import run as run_pipeline

    config_path = Path(__file__).parent.parent / "config.default.yaml"
    cfg = load_config(str(config_path) if config_path.exists() else None)
    packets = list(run_pipeline(str(COMEDY_MP4), cfg))
    return packets


# ---------------------------------------------------------------------------
# Test 1 — Static segment stays at low FPS
# ---------------------------------------------------------------------------

def test_static_segment_stays_low_fps():
    """
    During 00:00–00:30 (static/dialogue-heavy), non-scene-change packets must
    have fps_used <= 1.0. Scene-change packets are allowed to spike to 2–5 fps
    (that's correct adaptive behavior — genuine cuts deserve more frames).

    A violation (non-scene-change frame at 5+ fps) would indicate the motion or
    audio signal is spuriously high during static content.
    """
    _require_clip()
    packets = _run_pipeline()

    t_start, t_end = GROUND_TRUTH["static_segment_ms"]
    static_packets = [
        p for p in packets
        if t_start <= p.timestamp_ms <= t_end
    ]

    assert static_packets, "No packets emitted in static segment (0–30s)"

    # Allow scene-change packets to have higher fps — that's correct behavior.
    # Only non-scene-change packets should stay at <= 1.0 fps.
    non_scene_violators = [
        p for p in static_packets
        if not p.scene_change and p.fps_used > 1.0
    ]
    if non_scene_violators:
        details = [(round(p.timestamp_ms/1000, 1), p.fps_used, p.motion_score, p.decision_reason)
                   for p in non_scene_violators[:5]]
        pytest.fail(
            f"{len(non_scene_violators)} non-scene-change packets in static segment "
            f"exceeded 1.0 fps (spurious signal): {details}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Action segment reaches high FPS
# ---------------------------------------------------------------------------

def test_action_segment_reaches_high_fps():
    """
    During 00:31–01:00 (car roll / sprint / jump), at least one packet must
    have fps_used >= 5.0. If none do, the motion/scene signals are still too
    diluted to reach the 5-fps importance tier.
    """
    _require_clip()
    packets = _run_pipeline()

    t_start, t_end = GROUND_TRUTH["action_segment_ms"]
    action_packets = [
        p for p in packets
        if t_start <= p.timestamp_ms <= t_end
    ]

    assert action_packets, "No packets emitted in action segment (31–60s)"

    high_fps = [p for p in action_packets if p.fps_used >= 5.0]
    assert high_fps, (
        f"No packets in action segment (31–60s) reached fps_used >= 5.0. "
        f"Max fps_used seen: {max(p.fps_used for p in action_packets):.1f}. "
        f"Sample motion scores: {[round(p.motion_score, 3) for p in action_packets[:10]]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Scene cuts are detected
# ---------------------------------------------------------------------------

def test_scene_cuts_detected():
    """
    Each of the 4 known hard cuts must produce at least one packet with
    scene_change=True within ±700ms of the cut timestamp.

    700ms tolerance accounts for scheduler frame-selection granularity:
    at 30fps (33.3ms/frame), a cut landing on a non-emitted native frame
    can appear up to ~633ms late in the emitted packet stream.
    """
    _require_clip()
    packets = _run_pipeline()

    tolerance = GROUND_TRUTH["scene_cut_tolerance_ms"]
    known_cuts = GROUND_TRUTH["known_cuts_ms"]
    missed_cuts = []

    for cut_ms in known_cuts:
        window = [
            p for p in packets
            if abs(p.timestamp_ms - cut_ms) <= tolerance and p.scene_change
        ]
        if not window:
            missed_cuts.append(cut_ms)

    if missed_cuts:
        # For debugging: report all scene_change=True timestamps
        sc_true = [p.timestamp_ms for p in packets if p.scene_change]
        pytest.fail(
            f"scene_change=True not detected within ±{tolerance}ms of cuts at "
            f"{missed_cuts} ms. Detected scene changes at: {sc_true}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Novelty is not constant across the clip
# ---------------------------------------------------------------------------

def test_novelty_is_not_constant():
    """
    novelty_score must not be a constant value across all emitted packets.
    The v1.0 bug produced exactly 1.0 on every packet.
    """
    _require_clip()
    packets = _run_pipeline()

    assert packets, "No packets emitted — pipeline produced no output"

    scores = [p.novelty_score for p in packets]
    unique_values = set(round(s, 4) for s in scores)

    assert len(unique_values) > 1, (
        f"novelty_score is constant ({scores[0]:.4f}) across all {len(packets)} packets. "
        "Novelty saturation bug is still present."
    )

    # Additional: should not be uniformly 1.0 (the v1.0 symptom)
    all_ones = all(abs(s - 1.0) < 1e-6 for s in scores)
    assert not all_ones, (
        "novelty_score is 1.0 on every packet — the empty-buffer saturation bug is "
        "still present (fix: return 0.0 when buffer is empty, not 1.0)."
    )


# ---------------------------------------------------------------------------
# Test 5 — Runtime under threshold
# ---------------------------------------------------------------------------

def test_runtime_under_threshold():
    """
    Full pipeline on comedy.mp4 must complete in under 60 seconds.
    The v1.0 runtime of ~20 minutes was caused by audio re-decoding per frame.
    """
    _require_clip()
    from asvl.config import load_config
    from asvl.pipeline import run as run_pipeline

    config_path = Path(__file__).parent.parent / "config.default.yaml"
    cfg = load_config(str(config_path) if config_path.exists() else None)

    t0 = time.perf_counter()
    packets = list(run_pipeline(str(COMEDY_MP4), cfg))
    elapsed = time.perf_counter() - t0

    max_secs = GROUND_TRUTH["max_runtime_seconds"]
    assert elapsed < max_secs, (
        f"Pipeline took {elapsed:.1f}s — exceeded {max_secs}s limit. "
        f"Check per-stage timing in log output (decode/motion/scene/audio/novelty/schedule)."
    )
    assert packets, "No packets emitted despite pipeline completing within time."


# ---------------------------------------------------------------------------
# Unit test — motion score p90 vs old mean formula
# ---------------------------------------------------------------------------

def test_motion_score_p90_higher_than_mean_on_synthetic():
    """
    Regression unit test for the motion p90 fix (Fix 2).

    Synthetic setup: 1280×720 textured frame where a 250×350 px block (~10% of
    frame area) shifts 60px right against a random-noise background. Farneback
    requires texture to track. The 60px shift at 720p gets downsampled to ~6px
    at the 90px short-edge thumbnail, producing p90 magnitudes in the 5–8 px/frame
    range at thumbnail scale. With MOTION_NORM_CONST=10.0, this maps to 0.5–0.8.

    Contract assertions:
    1. New p90 score > 0.5 (fast-moving ~10% subject scores high).
    2. Old mean/50.0 score < 0.15 (same input saturated by background dilution).
    3. p90 score > 3× old mean score (substantial improvement).
    """
    import cv2
    from asvl.features.motion import compute_motion_score, MOTION_NORM_CONST

    rng = np.random.default_rng(42)
    H, W = 720, 1280

    # Base: random noise for trackable texture
    base = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    prev = base.copy()
    curr = base.copy()

    # Moving block: ~10% of frame area so it pushes p90 above threshold.
    # 250 rows × 350 cols = 87,500 px / (720 × 1280) ≈ 9.5% of frame area.
    # 60px shift at 720p → ~6px at the 90px short-edge thumbnail, producing
    # p90 magnitudes in the 5–8 px/frame range → score 0.5–0.8 at NORM_CONST=10.
    shift = 60
    prev[200:450, 300:650] = rng.integers(200, 256, (250, 350, 3), dtype=np.uint8)
    curr[200:450, 300 + shift:650 + shift] = prev[200:450, 300:650]
    # Erase old block location in curr to create a clean displacement
    curr[200:450, 300:300 + shift] = base[200:450, 300:300 + shift]

    score = compute_motion_score(prev, curr)

    # p90 score must be significantly higher than old mean-based score would have been
    assert score > 0.5, (
        f"p90-based motion score {score:.3f} too low for a 30px-shifted block. "
        "Expected > 0.5."
    )

    # Verify the old mean score would have been diluted (informational)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    old_mean_score = float(np.mean(magnitude) / 50.0)  # old formula

    assert old_mean_score < 0.15, (
        f"Baseline mean score {old_mean_score:.3f} unexpectedly high."
    )
    assert score > old_mean_score * 3, (
        f"p90 score {score:.3f} should be at least 3× the old mean score {old_mean_score:.3f}."
    )


# ---------------------------------------------------------------------------
# Unit test — novelty: cold-start returns 0.0, not 1.0
# ---------------------------------------------------------------------------

def test_novelty_cold_start_returns_zero():
    """Buffer is None → novelty must be 0.0, not 1.0."""
    from asvl.features.novelty import compute_novelty
    import numpy as np

    dummy_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    score = compute_novelty(dummy_frame, None)
    assert score == 0.0, f"Expected 0.0 for empty buffer, got {score}"


def test_novelty_identical_frames_low_score():
    """Near-identical frames should produce a low novelty score (< 0.2)."""
    from asvl.features.novelty import _frame_histogram, compute_novelty
    import numpy as np

    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    # Simulate a buffer mean hist from 10 identical frames
    buf_hist = _frame_histogram(frame)

    score = compute_novelty(frame, buf_hist)
    assert score < 0.2, f"Identical frame novelty {score:.3f} should be < 0.2"


def test_novelty_different_frame_high_score():
    """A genuinely different frame vs a buffer of uniform frames should produce high novelty (> 0.6)."""
    from asvl.features.novelty import _frame_histogram, compute_novelty
    import numpy as np

    # Buffer: solid grey frames
    ref_frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    buf_hist = _frame_histogram(ref_frame)

    # Query: completely different (saturated red)
    diff_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    diff_frame[:, :, 0] = 255  # pure red

    score = compute_novelty(diff_frame, buf_hist)
    assert score > 0.6, f"Different frame novelty {score:.3f} should be > 0.6"
