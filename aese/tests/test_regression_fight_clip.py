"""
tests/test_regression_fight_clip.py
Permanent regression fixture for the 81-second fight clip.

Background:
    A real run on a 2-character dialogue-to-fight video produced a single 81s event
    with importance=0.0844.  Root cause: scene_change and sustained motion were
    only weighted terms in the fused score, so they were diluted below the threshold
    by the quiet surrounding seconds.

    Fix: two hard-trigger checks in boundary/candidate_detector.py:
        1. camera_cue == "cut"          → is_boundary=True, confidence=0.95
        2. 2 consecutive fast_action    → is_boundary=True, confidence=0.85

Known structure of the failing clip (synthetic, no real video required):
    0 – 30s:  static / dialogue (motion_score ≈ 0.05)
    30 – 50s: fast_action / fight (motion_score ≈ 0.70, no scene_change)
    69s:      hard cut (scene_change=True)
    69 – 81s: static / end-cards (motion_score ≈ 0.02)

Definition of done:
    • At least one boundary fires within 2 seconds of the 30s action transition
    • A boundary fires at ~69s (±1000ms), tagged scene_change
    • No single event spans more than 40 000 ms (would indicate collapse)
    • Output uses character_count_range + max_characters_seen, never 'characters'
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Iterator, List

import numpy as np
import pytest

from aese.types import AESEConfig, Event, FramePacket


# ---------------------------------------------------------------------------
# Synthetic FramePacket stream helpers
# ---------------------------------------------------------------------------

def _make_packet(
    frame_id: int,
    timestamp_ms: float,
    motion_score: float,
    scene_change: bool,
    audio_energy: float = 0.05,
    novelty_score: float = 0.15,
    subtitle_text: str = None,
) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        fps_used=1.0,
        motion_score=motion_score,
        scene_change=scene_change,
        audio_energy=audio_energy,
        novelty_score=novelty_score,
        decision_reason="test",
        subtitle_text=subtitle_text,
        image=None,  # no real video -- image-dependent adapters get black frames
    )


def _fight_clip_stream() -> Iterator[FramePacket]:
    """
    Synthetic packet stream mirroring the known structure of the failing 81s clip.
    One FramePacket per second; all images are None (no real video required).
    """
    packets: List[FramePacket] = []

    # Segment 1: 0-29s -- static / dialogue
    for s in range(30):
        packets.append(_make_packet(
            frame_id=s,
            timestamp_ms=float(s * 1000),
            motion_score=0.05,
            scene_change=False,
            audio_energy=0.10,
            novelty_score=0.10,
            subtitle_text="Character A says something." if s % 3 == 0 else None,
        ))

    # Segment 2: 30-49s -- fast_action / fight
    # No scene_change -- Module 1 doesn't fire a cut here.
    # The hard-trigger should fire on 2 consecutive fast_action seconds.
    for s in range(30, 50):
        packets.append(_make_packet(
            frame_id=s,
            timestamp_ms=float(s * 1000),
            motion_score=0.70,
            scene_change=False,
            audio_energy=0.40,
            novelty_score=0.55,
        ))

    # Segment 3: 50-68s -- slow-down / recovery, still no cut
    for s in range(50, 69):
        packets.append(_make_packet(
            frame_id=s,
            timestamp_ms=float(s * 1000),
            motion_score=0.08,
            scene_change=False,
            audio_energy=0.08,
            novelty_score=0.12,
        ))

    # Second 69: hard cut to promo end-cards
    packets.append(_make_packet(
        frame_id=69,
        timestamp_ms=69000.0,
        motion_score=0.03,
        scene_change=True,   # Module 1 confirmed hard cut
        audio_energy=0.20,
        novelty_score=0.45,
    ))

    # Segment 4: 70-80s -- end-cards (static)
    for s in range(70, 81):
        packets.append(_make_packet(
            frame_id=s,
            timestamp_ms=float(s * 1000),
            motion_score=0.02,
            scene_change=False,
            audio_energy=0.05,
            novelty_score=0.08,
        ))

    return iter(packets)


def _run_pipeline_on_stream(stream) -> List[Event]:
    """Run the full AESE pipeline and collect all emitted events."""
    from aese.pipeline import run as aese_run
    config = AESEConfig()
    return list(aese_run(stream, config))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fight_clip_events() -> List[Event]:
    """
    Run the pipeline once on the synthetic fight clip and cache the results.
    All four tests share this run to avoid redundant processing.
    """
    return _run_pipeline_on_stream(_fight_clip_stream())


# ---------------------------------------------------------------------------
# Test 1 -- Action segment produces a boundary
# ---------------------------------------------------------------------------

def test_action_segment_produces_boundary(fight_clip_events):
    """
    AESE must emit at least one event boundary within a few seconds of the
    30s action transition (via the 2-consecutive-fast_action hard trigger).

    Failure here means the motion_spike hard trigger is not firing.
    """
    boundary_times_ms = [e.end_time_ms for e in fight_clip_events]

    # Hard trigger fires after 2 consecutive fast_action seconds.
    # First fast_action packet is at t=30s; trigger fires at t=32s (3rd fast_action).
    # Allow up to t=35s for timing tolerance.
    action_window_start = 30_000.0
    action_window_end   = 35_000.0

    found = any(action_window_start <= t <= action_window_end for t in boundary_times_ms)
    assert found, (
        f"No boundary in [{action_window_start / 1000:.0f}s, {action_window_end / 1000:.0f}s]. "
        f"Boundary times: {[t / 1000 for t in boundary_times_ms]}. "
        "The motion_spike hard trigger should fire within 2s of the 30s action onset."
    )


# ---------------------------------------------------------------------------
# Test 2 -- Hard cut at 69s produces a hard-triggered boundary
# ---------------------------------------------------------------------------

def test_known_cut_produces_hard_triggered_boundary(fight_clip_events):
    """
    AESE must emit an event boundary within +/-1000ms of the known hard cut at 69 000ms,
    with boundary_reason == 'scene_change' (from the camera-cut hard trigger).

    Failure here means the scene_change hard trigger is not firing, or the
    camera_cues adapter is not mapping scene_change=True to camera_cue='cut'.
    """
    cut_ms = 69_000.0
    tolerance_ms = 1_000.0

    cut_boundaries = [
        e for e in fight_clip_events
        if abs(e.end_time_ms - cut_ms) <= tolerance_ms
        and e.boundary_reason == "scene_change"
    ]

    assert len(cut_boundaries) >= 1, (
        f"No scene_change boundary within +/-{tolerance_ms / 1000:.1f}s of the 69s cut. "
        f"Events: "
        f"{[(e.start_time_ms / 1000, e.end_time_ms / 1000, e.boundary_reason) for e in fight_clip_events]}. "
        "The camera-cut hard trigger should have committed immediately at confidence=0.95."
    )


# ---------------------------------------------------------------------------
# Test 3 -- No single event spans the full clip (no collapse to one mega-event)
# ---------------------------------------------------------------------------

def test_no_event_spans_full_clip(fight_clip_events):
    """
    No single event should have duration_ms > 40 000ms (40s).

    The original defect: a single 81 000ms event with importance=0.0844.
    With hard triggers active, the dialogue / action / cut boundaries must each
    fire, producing at least 2 events each < 40s.
    """
    max_duration_ms = max(e.duration_ms for e in fight_clip_events)
    assert max_duration_ms <= 40_000.0, (
        f"Longest event is {max_duration_ms / 1000:.1f}s -- expected <= 40s. "
        f"Events: "
        f"{[(e.start_time_ms / 1000, e.end_time_ms / 1000, e.boundary_reason) for e in fight_clip_events]}. "
        "This is the original collapse defect: dialogue+action+cut merged into one mega-event."
    )

    assert len(fight_clip_events) >= 2, (
        f"Only {len(fight_clip_events)} event(s) produced -- expected at least 2. "
        "Hard triggers should have fired at the action transition and/or the 69s cut."
    )


# ---------------------------------------------------------------------------
# Test 4 -- character_count_range and max_characters_seen are present
# ---------------------------------------------------------------------------

def test_character_count_range_not_misleading(fight_clip_events):
    """
    Output events must have 'character_count_range' and 'max_characters_seen' fields.
    The old 'characters' field must not exist on the Event dataclass.

    Also verifies type contracts and internal consistency.
    """
    for event in fight_clip_events:
        assert hasattr(event, "character_count_range"), (
            "Event missing 'character_count_range' -- types.py rename incomplete."
        )
        assert hasattr(event, "max_characters_seen"), (
            "Event missing 'max_characters_seen' -- new field not added to types.py."
        )
        assert not hasattr(event, "characters"), (
            "Event still has old 'characters' field -- types.py rename not complete."
        )

        ccr = event.character_count_range
        mcs = event.max_characters_seen

        assert ccr is None or (
            isinstance(ccr, list) and all(isinstance(x, int) for x in ccr)
        ), f"character_count_range must be None or List[int], got {ccr!r}"

        assert mcs is None or isinstance(mcs, int), (
            f"max_characters_seen must be None or int, got {mcs!r}"
        )

        if ccr is not None and len(ccr) > 0:
            assert mcs is not None, (
                f"max_characters_seen is None but character_count_range={ccr!r} is non-empty"
            )
            assert mcs == max(ccr), (
                f"max_characters_seen={mcs} != max(character_count_range)={max(ccr)}"
            )
