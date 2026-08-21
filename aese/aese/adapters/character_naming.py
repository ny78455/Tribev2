"""
aese/adapters/character_naming.py
Subtitle-driven character naming for AESE.

Extracts spoken names from subtitle vocatives and binds them to face
clusters with strict evidence requirements. An unresolved "Person A" is
the correct output when evidence is absent; a wrong name on the wrong
face is a worse failure.

Guardrails:
  - Only HIGH-confidence vocatives (whole-line call-outs like "Dev!") trigger binding.
  - Only bind when EXACTLY ONE cluster is visible at the time of the call-out.
    Multiple visible clusters -> ambiguous -> silently discard.
  - Conflicting evidence (tied votes) leaves the cluster unresolved.
  - Retroactive relabeling is display-time ONLY; never feeds into online logic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocative pattern matching
# ---------------------------------------------------------------------------

@dataclass
class NameMention:
    """A candidate character name extracted from subtitle text."""
    name: str
    confidence: str    # "high" | "medium"
    timestamp_ms: float


# High confidence: entire subtitle line is just a name -- "Dev!" or "Manky."
_WHOLE_LINE_VOCATIVE = re.compile(r"^([A-Z][a-zA-Z]{1,20})[!.]?$")

# Medium confidence: comma-vocative at end of sentence -- "Are you home, Dev?"
_COMMA_VOCATIVE = re.compile(r",\s*([A-Z][a-zA-Z]{1,20})[?.!]?\s*$")

# Stopwords: common capitalized non-name words to suppress false positives.
_STOPWORDS = {
    "Why", "Are", "Both", "Very", "Come", "That", "This", "What",
    "Daughter", "Ciao", "Ok", "Okay", "Yes", "No", "Please", "Sorry",
    "Hey", "Oh", "Ah", "Wait", "Look", "Listen", "Fine", "Good",
    "Stop", "Run", "Go", "Help", "Now", "Just", "Then", "Sir",
    "Miss", "Mister", "Mr", "Mrs", "Dr", "Phone", "Dear", "Shut",
}


def extract_name_mentions(subtitle_text: str, timestamp_ms: float) -> List[NameMention]:
    """
    Extract potential vocative name mentions from a subtitle line.

    Only returns "high" or "medium" confidence mentions. Low-confidence
    mid-sentence third-party references ("Tell Sangit...") are excluded --
    there is no reliable on-screen attribution from them alone.
    """
    if not subtitle_text:
        return []
    text = subtitle_text.strip()
    mentions: List[NameMention] = []

    # Whole-line vocative -- highest confidence; return immediately
    m = _WHOLE_LINE_VOCATIVE.match(text)
    if m and m.group(1) not in _STOPWORDS:
        mentions.append(NameMention(name=m.group(1), confidence="high", timestamp_ms=timestamp_ms))
        return mentions

    # Comma-vocative -- medium confidence
    m = _COMMA_VOCATIVE.search(text)
    if m and m.group(1) not in _STOPWORDS:
        mentions.append(NameMention(name=m.group(1), confidence="medium", timestamp_ms=timestamp_ms))

    return mentions


# ---------------------------------------------------------------------------
# Evidence accumulator + cluster binder
# ---------------------------------------------------------------------------

@dataclass
class ClusterNameEvidence:
    """Accumulated name votes for one face cluster."""
    votes: Dict[str, int] = field(default_factory=dict)
    resolved_name: Optional[str] = None


class CharacterNameBinder:
    """
    Accumulates subtitle vocative evidence and resolves face cluster -> real
    name bindings only when evidence is strong and unambiguous.

    Rules:
      1. Only HIGH-confidence mentions contribute (medium alone is too weak).
      2. Only accumulate when EXACTLY ONE cluster is visible.
      3. Resolve only when the top name beats ALL rivals by vote count.
      4. Tied / conflicting evidence -> leave unresolved.

    Args:
        min_votes_to_resolve: Corroborating observations needed before
            committing. Default 1 (first clean evidence is enough).
    """

    def __init__(self, min_votes_to_resolve: int = 1) -> None:
        self.evidence: Dict[str, ClusterNameEvidence] = {}
        self.min_votes = min_votes_to_resolve

    def observe(self, mention: NameMention, active_cluster_labels: List[str]) -> None:
        """
        Record a name mention with the set of visible clusters at that moment.
        Silently discards medium-confidence and ambiguous multi-cluster cases.
        """
        if mention.confidence != "high":
            return
        if len(active_cluster_labels) != 1:
            return  # Ambiguous -- multiple faces, cannot attribute

        cluster = active_cluster_labels[0]
        ev = self.evidence.setdefault(cluster, ClusterNameEvidence())
        ev.votes[mention.name] = ev.votes.get(mention.name, 0) + 1

        top_name = max(ev.votes, key=ev.votes.get)
        top_count = ev.votes[top_name]
        rival_total = sum(v for n, v in ev.votes.items() if n != top_name)

        if top_count >= self.min_votes and top_count > rival_total:
            if ev.resolved_name != top_name:
                ev.resolved_name = top_name
                logger.info(
                    "AESE naming: cluster %r resolved to %r (votes=%d)",
                    cluster, top_name, top_count,
                )
        else:
            # Conflicting -- revert rather than commit a wrong name
            if ev.resolved_name is not None:
                logger.info("AESE naming: cluster %r became conflicted -- unresolved", cluster)
            ev.resolved_name = None

    def resolved_names(self) -> Dict[str, str]:
        """Return clusters with confirmed bindings only (omits unresolved)."""
        return {
            c: ev.resolved_name
            for c, ev in self.evidence.items()
            if ev.resolved_name is not None
        }


# ---------------------------------------------------------------------------
# Batch display-time retroactive relabeling
# ---------------------------------------------------------------------------

def apply_resolved_names(events: List["Event"], binder: CharacterNameBinder) -> None:
    """
    Replace anonymous cluster labels with resolved real names across ALL events.

    BATCH POST-PROCESS only -- runs after the full clip is processed. Applies
    retroactively: an event at t=2s gets the name evidenced at t=15s because
    they are the same physical person. Display-time only; the online boundary /
    importance logic is never affected.
    """
    resolved = binder.resolved_names()
    if not resolved:
        return
    logger.info("AESE naming: applying resolved names to %d events: %s", len(events), resolved)
    for event in events:
        event.character_labels = [resolved.get(label, label) for label in event.character_labels]
        for old_label, new_name in resolved.items():
            if old_label in event.summary:
                event.summary = event.summary.replace(old_label, new_name)
