"""
tests/test_render.py
Tests for the human-readable event log renderer (aese/render.py).

Covers:
  - format_timestamp correctness at edge cases
  - render_event_log structure (per spec Section 27)
  - character label fallback to placeholder when character_labels absent
  - location_label None -> "Unknown"
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from aese.render import format_timestamp, render_event_log
from aese.types import Event


def _make_event(
    event_id=0,
    start_ms=0.0,
    end_ms=105000.0,
    event_type="Dialogue",
    summary="Two people talk.",
    location_label=None,
    max_characters_seen=None,
    character_labels=None,
) -> Event:
    """Minimal Event factory for renderer tests."""
    e = Event(
        event_id=event_id,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        duration_ms=end_ms - start_ms,
        event_embedding=np.zeros(4, dtype=np.float32),
        importance=0.5,
        confidence=0.8,
        summary=summary,
        boundary_reason="scene",
        event_type=event_type,
        location_label=location_label,
        max_characters_seen=max_characters_seen,
    )
    if character_labels is not None:
        e.character_labels = character_labels
    return e


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------

def test_format_timestamp_zero():
    assert format_timestamp(0) == "00:00:00"


def test_format_timestamp_one_second():
    assert format_timestamp(1000) == "00:00:01"


def test_format_timestamp_one_minute():
    assert format_timestamp(60000) == "00:01:00"


def test_format_timestamp_hour():
    # 3661 seconds = 1h 1m 1s
    assert format_timestamp(3661000) == "01:01:01"


def test_format_timestamp_large():
    # 2h 30m 15s = 9015 seconds
    assert format_timestamp(9015000) == "02:30:15"


def test_format_timestamp_sub_second_truncated():
    # 500ms -> rounds down to 0 seconds
    assert format_timestamp(500) == "00:00:00"


# ---------------------------------------------------------------------------
# render_event_log -- structure
# ---------------------------------------------------------------------------

def test_render_event_log_empty_returns_empty():
    assert render_event_log([]) == ""


def test_render_event_log_structure_one_event():
    """Single event must contain all six required fields in the right order."""
    e = _make_event(
        event_id=0,
        start_ms=0.0,
        end_ms=105000.0,  # 00:01:45
        event_type="Dialogue",
        summary="Two people talk across a desk.",
        location_label="office",
        max_characters_seen=2,
        character_labels=["Person A", "Person B"],
    )
    text = render_event_log([e])
    assert "Event 001" in text
    assert "00:00:00 -> 00:01:45" in text
    assert "Type: Dialogue" in text
    assert "Characters: Person A, Person B" in text
    assert "Location: Office" in text or "Location: office" in text
    assert "Summary: Two people talk across a desk." in text


def test_render_event_log_two_events_separated_by_blank():
    """Multiple events must be separated by a blank line."""
    e1 = _make_event(event_id=0, start_ms=0.0, end_ms=10000.0)
    e2 = _make_event(event_id=1, start_ms=10000.0, end_ms=25000.0, event_type="Action")
    text = render_event_log([e1, e2])
    assert "Event 001" in text
    assert "Event 002" in text
    # Blank line separates them
    assert "\n\n" in text


def test_render_event_log_sequential_numbering():
    """Events must be numbered 001, 002, 003 regardless of event_id."""
    events = [_make_event(event_id=i, start_ms=i*10000.0, end_ms=(i+1)*10000.0) for i in range(3)]
    text = render_event_log(events)
    assert "Event 001" in text
    assert "Event 002" in text
    assert "Event 003" in text


# ---------------------------------------------------------------------------
# render_event_log -- character label fallback
# ---------------------------------------------------------------------------

def test_render_event_log_placeholder_chars_from_max_seen():
    """
    When character_labels is absent/empty but max_characters_seen=2,
    the renderer must use 'Person 1, Person 2' as placeholder.
    """
    e = _make_event(max_characters_seen=2, character_labels=[])
    text = render_event_log([e])
    assert "Person 1" in text
    assert "Person 2" in text


def test_render_event_log_no_chars():
    """When max_characters_seen is None and no labels, display 'None'."""
    e = _make_event(max_characters_seen=None, character_labels=[])
    text = render_event_log([e])
    assert "Characters: None" in text


def test_render_event_log_char_labels_override_placeholder():
    """Fix 4 labels take priority over max_characters_seen placeholder."""
    e = _make_event(max_characters_seen=5, character_labels=["Person A"])
    text = render_event_log([e])
    assert "Person A" in text
    assert "Person 2" not in text  # placeholder not used


# ---------------------------------------------------------------------------
# render_event_log -- location fallback
# ---------------------------------------------------------------------------

def test_render_event_log_none_location_displays_unknown():
    e = _make_event(location_label=None)
    text = render_event_log([e])
    assert "Location: Unknown" in text


def test_render_event_log_real_location():
    e = _make_event(location_label="kitchen")
    text = render_event_log([e])
    # capitalize() makes "kitchen" -> "Kitchen"
    assert "Location: Kitchen" in text