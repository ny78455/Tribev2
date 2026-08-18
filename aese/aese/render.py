"""
aese/render.py
Human-readable event log renderer.

Produces the narrative event-log format from the source spec Section 27:

    Event 001
    00:00:00 -> 00:01:45
    Type: Dialogue
    Characters: Person A, Person B
    Location: office
    Summary: Two people discuss a contract across a desk.

This is a RENDERING layer on top of the canonical JSONL output.
JSONL stays the machine-readable source of truth; this renderer
reads structured Event objects and produces a human-legible log.

Usage:
    from aese.render import render_event_log
    text = render_event_log(events)

    # Or from CLI:
    python cli.py --input manifest.jsonl --output events.jsonl --format human
    # Produces events.txt alongside events.jsonl
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .types import Event


def format_timestamp(ms: float) -> str:
    """
    Convert milliseconds to HH:MM:SS string.

    Args:
        ms: Time in milliseconds (non-negative).

    Returns:
        str: Zero-padded HH:MM:SS string, e.g. "01:23:45".
    """
    total_seconds = int(ms / 1000)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_character_display(event: "Event") -> str:
    """
    Return a human-readable character string for the event.

    Priority:
      1. event.character_labels (Fix 4 anonymous clustering or Fix 5 named refs)
      2. Placeholder "Person N" list derived from max_characters_seen
         (used when Fix 4 has not yet run, e.g. manifest-replay without --video)
      3. "None" if no character data is available at all.
    """
    # Fix 4/5 path: structured labels available
    labels = getattr(event, "character_labels", None)
    if labels:
        return ", ".join(labels)

    # Placeholder path: only a count is available
    max_seen = getattr(event, "max_characters_seen", None)
    if max_seen is not None and max_seen > 0:
        return ", ".join(f"Person {i + 1}" for i in range(max_seen))

    # No character data available (manifest-replay, image=None)
    return "None"


def render_event_log(events: List["Event"]) -> str:
    """
    Render a list of finalized Events as a human-readable event log.

    Format per spec Section 27:
        Event 001
        HH:MM:SS -> HH:MM:SS
        Type: <event_type>
        Characters: <labels or placeholder>
        Location: <location_label or Unknown>
        Summary: <summary>

    Each event is separated by a blank line. The final block has no trailing
    blank line.

    Args:
        events: List of finalized Event objects (output of pipeline.run()).

    Returns:
        str: Human-readable event log. Empty string if events list is empty.
    """
    if not events:
        return ""

    blocks = []
    for i, event in enumerate(events, start=1):
        start_ts = format_timestamp(event.start_time_ms)
        end_ts = format_timestamp(event.end_time_ms)
        chars = _resolve_character_display(event)
        location = (event.location_label or "Unknown").capitalize()
        summary = event.summary or "(no summary)"

        block = (
            f"Event {i:03d}\n"
            f"{start_ts} -> {end_ts}\n"
            f"Type: {event.event_type}\n"
            f"Characters: {chars}\n"
            f"Location: {location}\n"
            f"Summary: {summary}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)