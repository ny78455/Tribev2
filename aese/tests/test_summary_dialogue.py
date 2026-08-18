"""
tests/test_summary_dialogue.py
Tests for dialogue-grounded narrative summaries (Fix 6).

Covers:
  - Dialogue context injected into VLM prompt when event.dialogue_text is set
  - No injection when dialogue_text is None/empty
  - Filler validation still rejects bad output even with dialogue context
  - Template fallback when VLM unavailable (unchanged behavior)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest.mock as mock
import numpy as np
import pytest

from aese.summary import generate_summary, _validate_or_fallback, _summary_call_counter
from aese.types import Event


def _make_event(summary="", event_type="Dialogue", location_label="office",
                max_characters_seen=2, dialogue_text=None) -> Event:
    e = Event(
        event_id=0,
        start_time_ms=0.0,
        end_time_ms=10000.0,
        duration_ms=10000.0,
        event_embedding=np.zeros(4, dtype=np.float32),
        importance=0.5,
        confidence=0.8,
        summary=summary,
        boundary_reason="scene",
        event_type=event_type,
        location_label=location_label,
        max_characters_seen=max_characters_seen,
    )
    e.dialogue_text = dialogue_text
    return e


_DUMMY_IMAGE = np.full((8, 8, 3), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dialogue injection
# ---------------------------------------------------------------------------

def test_dialogue_injected_into_prompt():
    """When event.dialogue_text is non-empty, the VLM prompt must include it."""
    captured_prompts = []

    def mock_call_vlm(system_prompt, image, max_tokens=60):
        captured_prompts.append(system_prompt)
        return "A person sits at a desk and says they have to go."

    event = _make_event(dialogue_text="I have to go now.")
    with mock.patch("aese.summary._vlm_available", return_value=True):
        with mock.patch("aese.summary._call_vlm", side_effect=mock_call_vlm):
            _summary_call_counter.reset()
            result = generate_summary(event, _DUMMY_IMAGE)

    assert len(captured_prompts) == 1
    assert "I have to go now." in captured_prompts[0], (
        f"Dialogue not found in prompt: {captured_prompts[0]!r}"
    )
    assert "Dialogue spoken during this event" in captured_prompts[0]


def test_no_dialogue_no_injection():
    """When dialogue_text is None, the prompt must NOT contain 'Dialogue spoken'."""
    captured_prompts = []

    def mock_call_vlm(system_prompt, image, max_tokens=60):
        captured_prompts.append(system_prompt)
        return "A quiet room with a window."

    event = _make_event(dialogue_text=None)
    with mock.patch("aese.summary._vlm_available", return_value=True):
        with mock.patch("aese.summary._call_vlm", side_effect=mock_call_vlm):
            _summary_call_counter.reset()
            result = generate_summary(event, _DUMMY_IMAGE)

    assert len(captured_prompts) == 1
    assert "Dialogue spoken" not in captured_prompts[0], (
        f"Unexpected dialogue injection in prompt without dialogue_text: {captured_prompts[0]!r}"
    )


def test_empty_dialogue_no_injection():
    """Empty string dialogue_text is treated the same as None -- no injection."""
    captured_prompts = []

    def mock_call_vlm(system_prompt, image, max_tokens=60):
        captured_prompts.append(system_prompt)
        return "A hallway with fluorescent lights."

    event = _make_event(dialogue_text="")
    with mock.patch("aese.summary._vlm_available", return_value=True):
        with mock.patch("aese.summary._call_vlm", side_effect=mock_call_vlm):
            _summary_call_counter.reset()
            generate_summary(event, _DUMMY_IMAGE)

    assert "Dialogue spoken" not in captured_prompts[0]


# ---------------------------------------------------------------------------
# Filler still rejected with dialogue context
# ---------------------------------------------------------------------------

def test_filler_still_rejected_with_dialogue():
    """
    Even with dialogue context injected, if VLM produces filler text,
    _validate_or_fallback must reject it and return the template fallback.
    """
    filler_response = "Let me know if you need any further analysis."
    event = _make_event(dialogue_text="I have to go now.")
    with mock.patch("aese.summary._vlm_available", return_value=True):
        with mock.patch("aese.summary._call_vlm", return_value=filler_response):
            _summary_call_counter.reset()
            result = generate_summary(event, _DUMMY_IMAGE)

    # Must NOT be the filler response
    assert "Let me know" not in result, f"Filler was not rejected: {result!r}"
    # Must be the template fallback
    assert "office" in result or "Dialogue" in result, (
        f"Expected template fallback, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Template fallback when VLM unavailable (unchanged from prior contract)
# ---------------------------------------------------------------------------

def test_template_fallback_when_vlm_unavailable():
    """Without VLM, generate_summary must return template string regardless of dialogue."""
    event = _make_event(dialogue_text="This should not appear in the output.", location_label="kitchen")
    with mock.patch("aese.summary._vlm_available", return_value=False):
        _summary_call_counter.reset()
        result = generate_summary(event, _DUMMY_IMAGE)
    assert "Dialogue" in result or "kitchen" in result, (
        f"Expected template summary, got: {result!r}"
    )
    assert "This should not appear" not in result