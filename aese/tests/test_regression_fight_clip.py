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
            subtitle_text="Character A says something." if s % 2 == 0 else None,
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


# ---------------------------------------------------------------------------
# Test 5 -- Event IDs are contiguous
# ---------------------------------------------------------------------------

def test_event_ids_contiguous(fight_clip_events):
    assert [e.event_id for e in fight_clip_events] == list(range(len(fight_clip_events))), (
        f"Event IDs are not contiguous: {[e.event_id for e in fight_clip_events]}"
    )


# ---------------------------------------------------------------------------
# Test 6 -- No gaps between events
# ---------------------------------------------------------------------------

def test_no_gaps_between_events(fight_clip_events):
    for i in range(len(fight_clip_events) - 1):
        assert fight_clip_events[i].end_time_ms == fight_clip_events[i+1].start_time_ms, (
            f"Gap between event {fight_clip_events[i].event_id} (ends {fight_clip_events[i].end_time_ms}) and "
            f"event {fight_clip_events[i+1].event_id} (starts {fight_clip_events[i+1].start_time_ms})"
        )


# ---------------------------------------------------------------------------
# Test 7 -- Summary and max_characters_seen agree
# ---------------------------------------------------------------------------

def test_summary_and_max_characters_agree(fight_clip_events):
    for e in fight_clip_events:
        if e.max_characters_seen:
            assert str(e.max_characters_seen) in e.summary, (
                f"Summary '{e.summary}' does not contain expected count {e.max_characters_seen}"
            )


# ---------------------------------------------------------------------------
# Test 8 -- Action event outranks dialogue in importance
# ---------------------------------------------------------------------------

def test_action_event_outranks_dialogue_in_importance(fight_clip_events):
    action_events = [e for e in fight_clip_events if e.event_type == "Action"]
    dialogue_events = [e for e in fight_clip_events if e.event_type == "Dialogue"]
    
    assert action_events, "Expected at least one Action event"
    assert dialogue_events, "Expected at least one Dialogue event"
    
    max_action_importance = max(e.importance for e in action_events)
    max_dialogue_importance = max(e.importance for e in dialogue_events)
    
    assert max_action_importance > max_dialogue_importance, (
        f"Action event importance {max_action_importance:.4f} does not strictly outrank "
        f"Dialogue importance {max_dialogue_importance:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 9 -- No conversational filler in any summary (Fix 3 regression guard)
# ---------------------------------------------------------------------------

def test_no_conversational_filler_in_any_summary(fight_clip_events):
    """
    No event summary may contain conversational filler or bare markdown dividers.

    This guards against the regression where the generative VLM was producing
    outputs like "Let me know if you need anything else.\nAnswer" or "---\n---".
    The _validate_or_fallback gate in summary.py must catch all such outputs
    and substitute the template fallback before any event is emitted.

    In this test suite, VLM is unavailable (no model installed), so all summaries
    should be template-based and trivially clean. The test is still valuable as
    a canary: if filler somehow appears, a VLM has re-entered the hot path
    without going through the validation gate.
    """
    import re

    _FILLER_PATTERNS = [
        re.compile(r"(?i)let\s+me\s+know\s+if"),
        re.compile(r"(?i)i('| a)ll\s+be\s+happy\s+to"),
        re.compile(r"(?im)^(answer|here\s+is|here'?s)\b"),
        re.compile(r"(?m)^-{2,}\s*$"),
        re.compile(r"(?i)as\s+an\s+ai"),
        re.compile(r"(?i)i\s+(can'?t|cannot|am\s+unable)"),
    ]

    for e in fight_clip_events:
        assert e.summary, f"Event {e.event_id} has an empty summary -- must always have at least a template"
        for pattern in _FILLER_PATTERNS:
            assert not pattern.search(e.summary), (
                f"Event {e.event_id}: filler pattern {pattern.pattern!r} found in "
                f"summary {e.summary!r}. The _validate_or_fallback gate did not fire."
            )


# ---------------------------------------------------------------------------
# Test 10 -- Character counts not universally zero (Fix 1 regression guard)
# ---------------------------------------------------------------------------

def test_character_counts_not_universally_zero_or_all_none(fight_clip_events):
    """
    The synthetic fight clip has image=None for all packets (manifest-replay mode),
    so character_count_range and max_characters_seen should be None (not 0) for
    all events -- None means 'no image data available', not 'zero people seen'.

    This test verifies:
    1. max_characters_seen is None (not 0) when no real images were provided.
       A value of 0 in this mode would mean the detector was called on a black
       frame and returned 0 -- which is semantically wrong (should be None).
    2. The character_data_available flag correctly reflects the image availability.

    NOTE: If this test is run with real images (live mode), assert at least one
    event has max_characters_seen > 0. The synthetic stream uses image=None, so
    we assert None (not 0) as the correct sentinel for missing image data.
    """
    has_any_real_image = any(
        e.character_data_available and e.max_characters_seen is not None
        for e in fight_clip_events
    )

    if not has_any_real_image:
        # Manifest-replay mode: all images were None. Verify None sentinel, NOT 0.
        for e in fight_clip_events:
            assert e.max_characters_seen is None, (
                f"Event {e.event_id}: max_characters_seen={e.max_characters_seen} but "
                f"no real images were in the stream. Expected None (not 0). "
                f"A value of 0 suggests the detector ran on a black frame and returned "
                f"0 instead of being skipped. Check aggregator.py's None-guard."
            )
    else:
        # Live mode with real images: at least one event must have seen people.
        assert any(e.max_characters_seen and e.max_characters_seen > 0 for e in fight_clip_events), (
            "All events report max_characters_seen=0 with real images present. "
            "This is the free-text VLM parsing regression: count_people() returned "
            "filler text with no parseable integer, silently defaulting to 0. "
            "Fix: restore the deterministic OpenCV-only path in character_stub.py."
        )


# ---------------------------------------------------------------------------
# Test 11 -- generate_summary call count equals event count (Fix 2 + 3 guard)
# ---------------------------------------------------------------------------

def test_generative_summary_call_count_matches_event_count():
    """
    generate_summary() must be called exactly once per finalized event --
    never once per TemporalFeature, never once per boundary-decision step.

    This guards against the hot-path regression: calling VLM summary generation
    ~81 times for an 81s clip instead of ~8 times (once per event).

    Method: reset the _summary_call_counter, run the full pipeline on a fresh
    stream, then assert counter == len(events).
    """
    from aese.summary import _summary_call_counter
    from aese.pipeline import run as aese_run

    _summary_call_counter.reset()

    config = AESEConfig()
    events = list(aese_run(_fight_clip_stream(), config))

    assert _summary_call_counter.count == len(events), (
        f"generate_summary() was called {_summary_call_counter.count} times "
        f"but {len(events)} events were emitted. "
        f"Expected 1 call per finalized event. "
        f"If count >> len(events), a VLM summary call has re-entered the hot path "
        f"and is running once per TemporalFeature or per boundary-decision step."
    )
