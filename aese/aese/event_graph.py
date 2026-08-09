"""
aese/event_graph.py
Event Graph — §5.15, §24.

Structural event graph: nodes = Events, edges = "before" (temporal order only).

DELIBERATELY NOT IMPLEMENTED:
  "causes" edges — causal inference between events requires LLM reasoning or
  a trained causal model. This is explicitly out of scope for Module 2.
  Returning None or omitting that edge type entirely, per §5.15.
  See README.md and DECISIONS.md §10.

The graph provides:
  - Topological order of events
  - "before" adjacency (Event A is before Event B)
  - Node lookup by event_id
  - JSON serialization (without embeddings/keyframes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .types import Event


@dataclass
class EventNode:
    """A node in the EventGraph, wrapping an Event."""
    event: Event
    before: List[int] = field(default_factory=list)  # event_ids of events that come AFTER this
    # 'causes' edge intentionally omitted — out of scope; see DECISIONS.md §10


class EventGraph:
    """
    Directed temporal graph of events.

    Edges:
      before(A, B): event A ends before event B starts (temporal order).

    NOT implemented:
      causes(A, B): A caused B — requires causal reasoning; future work.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, EventNode] = {}  # event_id → EventNode
        self._order: List[int] = []             # event_ids in temporal order

    def add_event(self, event: Event) -> None:
        """Add an event node. Automatically creates a 'before' edge from the previous event."""
        node = EventNode(event=event)

        # Create 'before' edge from the last event to this one
        if self._order:
            prev_id = self._order[-1]
            self._nodes[prev_id].before.append(event.event_id)

        self._nodes[event.event_id] = node
        self._order.append(event.event_id)

    def get_event(self, event_id: int) -> Optional[Event]:
        """Look up an event by ID."""
        node = self._nodes.get(event_id)
        return node.event if node else None

    def events_in_order(self) -> List[Event]:
        """Return all events in temporal order."""
        return [self._nodes[eid].event for eid in self._order]

    def edges_before(self) -> List[Tuple[int, int]]:
        """Return all (a_id, b_id) 'before' edge pairs."""
        edges = []
        for eid in self._order:
            node = self._nodes[eid]
            for next_id in node.before:
                edges.append((eid, next_id))
        return edges

    def causes_edges(self) -> List:
        """
        'causes' edges — NOT IMPLEMENTED.
        Returns an empty list. Causal inference is out of scope for Module 2.
        See DECISIONS.md §10.
        """
        return []  # intentionally empty

    def to_json_safe(self) -> dict:
        """Serialize graph structure (no embedding arrays)."""
        nodes = []
        for eid in self._order:
            node = self._nodes[eid]
            ev = node.event
            nodes.append({
                "event_id": ev.event_id,
                "start_time_ms": ev.start_time_ms,
                "end_time_ms": ev.end_time_ms,
                "event_type": ev.event_type,
                "summary": ev.summary,
                "before": node.before,
                "causes": [],  # always empty; see DECISIONS.md §10
            })
        return {"events": nodes, "edges_before": self.edges_before()}

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events_in_order())
