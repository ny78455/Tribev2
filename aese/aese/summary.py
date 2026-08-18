"""
aese/summary.py
Post-finalization event summary generator.

Architectural contract (see DECISIONS.md §17):
  - generate_summary() is called ONCE PER FINALIZED EVENT in pipeline.py,
    after merge_events() / filter_short_events() / EventClassifier have all resolved.
  - It is NEVER called per TemporalFeature or per boundary-detection step.
  - The <100ms/decision hot-path constraint applies to the boundary loop; this
    function is explicitly outside that loop.

Output guarantees:
  - Always returns a non-empty string.
  - If the generative VLM path produces filler text, multi-line rambling, or any
    exception, the guaranteed fallback is build_template_summary(), which never fails.
  - Filler detection is performed against _FILLER_PATTERNS before any VLM output
    is accepted.

Testing hook:
  - _summary_call_counter.count tracks the number of generate_summary() calls.
  - _summary_call_counter.reset() resets it between test runs.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .types import Event


# ---------------------------------------------------------------------------
# Filler-pattern registry
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    re.compile(r"(?i)let\s+me\s+know\s+if"),
    re.compile(r"(?i)i('| a)ll\s+be\s+happy\s+to"),
    re.compile(r"(?im)^(answer|here\s+is|here'?s)\b"),
    re.compile(r"(?m)^-{2,}\s*$"),       # bare markdown dividers on their own line
    re.compile(r"(?i)as\s+an\s+ai"),
    re.compile(r"(?i)i\s+(can'?t|cannot|am\s+unable)"),
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "You are a factual video-event data extractor, not a conversational assistant. "
    "Given one representative frame from a video event, output EXACTLY ONE sentence "
    "describing what is visible: setting, subjects, and action. "
    "Do not include greetings, offers of further help, meta-commentary, or markdown. "
    "Do not say anything except the single descriptive sentence."
)


# ---------------------------------------------------------------------------
# Call counter -- testability hook
# ---------------------------------------------------------------------------

class _CallCounter:
    """Simple call counter for test instrumentation."""
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count: int = 0

    def increment(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


_summary_call_counter = _CallCounter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vlm_available() -> bool:
    """Return True if FastVLM is loaded and usable."""
    try:
        from .adapters.fastvlm import _fastvlm_available
        return bool(_fastvlm_available)
    except Exception:
        return False


def _call_vlm(system_prompt: str, image: np.ndarray, max_tokens: int = 60) -> str:
    """
    Thin wrapper around fastvlm._ask().
    Returns the raw response string, or "" on any failure.
    The system prompt is prepended to the user turn (standard approach for
    instruction-following with this model family).
    """
    try:
        from .adapters.fastvlm import _ask
        combined_prompt = f"{system_prompt}\n\nDescribe this frame."
        return _ask(image, combined_prompt, max_new_tokens=max_tokens)
    except Exception as exc:
        logger.debug("AESE summary: VLM call failed: %s", exc)
        return ""


def _validate_or_fallback(raw: str, fallback: str) -> str:
    """
    Validate VLM output against quality gates; return fallback if any gate fails.

    Gates (in order):
      1. Non-empty and at least 5 characters.
      2. No filler patterns from _FILLER_PATTERNS.
      3. No more than one newline (multi-line = rambling).

    Args:
        raw:      Raw VLM response string.
        fallback: Template summary -- guaranteed non-empty.

    Returns:
        The cleaned raw string if all gates pass, otherwise fallback.
    """
    cleaned = raw.strip()

    # Gate 1: length
    if not cleaned or len(cleaned) < 5:
        logger.debug("AESE summary: VLM output too short (%d chars) -- using fallback", len(cleaned))
        return fallback

    # Gate 2: filler patterns
    for pattern in _FILLER_PATTERNS:
        if pattern.search(cleaned):
            logger.debug(
                "AESE summary: filler pattern %r matched -- using fallback",
                pattern.pattern,
            )
            return fallback

    # Gate 3: multi-line rambling
    if cleaned.count("\n") > 1:
        logger.debug("AESE summary: multi-line VLM output -- using fallback")
        return fallback

    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(event: "Event", keyframe_image: Optional[np.ndarray]) -> str:
    """
    Generate a summary for a finalized event.

    Called ONCE PER FINALIZED EVENT in pipeline.py, never in the per-second
    hot path. Call count is tracked by _summary_call_counter.

    Architecture:
      1. Build the guaranteed template fallback first (always available).
      2. If no real image or VLM unavailable -> return template immediately.
      3. Call VLM with strict system prompt (max_tokens=60).
      4. Validate output through _validate_or_fallback.
      5. Return validated output or template fallback.

    Args:
        event:          Finalized Event object (post-merge/classify).
        keyframe_image: Representative RGB frame (np.ndarray) or None.

    Returns:
        str: Non-empty summary string (never filler, never multi-line).
    """
    _summary_call_counter.increment()

    # Build template fallback first -- always valid regardless of VLM availability.
    from .event_constructor import build_template_summary
    scene = event.location_label or "unknown"
    template_fallback = build_template_summary(
        event_type=event.event_type,
        scene_label=scene,
        max_characters_seen=event.max_characters_seen,
    )

    # Skip VLM if no real image data is available
    if keyframe_image is None or (hasattr(keyframe_image, "max") and keyframe_image.max() < 5):
        logger.debug("AESE summary: no real image -- using template for event %d", event.event_id)
        return template_fallback

    # Skip VLM if model not loaded
    if not _vlm_available():
        logger.debug("AESE summary: VLM unavailable -- using template for event %d", event.event_id)
        return template_fallback

    try:
        raw = _call_vlm(SUMMARY_SYSTEM_PROMPT, keyframe_image, max_tokens=60)
        result = _validate_or_fallback(raw, template_fallback)
        if result != template_fallback:
            logger.debug(
                "AESE summary: VLM summary accepted for event %d (%d chars)",
                event.event_id, len(result),
            )
        return result
    except Exception as exc:
        logger.warning(
            "AESE summary: unexpected error for event %d: %s -- using template",
            event.event_id, exc,
        )
        return template_fallback