"""
aese/context_buffer.py
Temporal Context Buffer — rolling deque of TemporalFeature records.

Holds up to buffer_seconds (default 45) of TemporalFeatures.
This is larger than Module 1's 10s rolling frame buffer because event coherence
requires more context than adaptive frame sampling decisions do:
  - A 45s window can hold a complete short scene (dialogue → action → resolution)
  - 10s is often smaller than a single conversation beat
See DECISIONS.md §7.

Exposed API:
  .push(tf)             → append a new TemporalFeature
  .mean_embedding()     → running mean of all embeddings in buffer
  .recent(n)            → last n TemporalFeatures
  .recent_embeddings(n) → stacked embedding matrix from last n features
  .last_boundary_time() → timestamp_ms of most recently recorded boundary
  .record_boundary(ts)  → mark a boundary timestamp
  .size                 → current number of features in buffer
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .types import TemporalFeature


class ContextBuffer:
    """Rolling buffer of TemporalFeatures with incremental mean embedding."""

    def __init__(self, buffer_seconds: float = 45.0) -> None:
        self._maxlen = math.ceil(buffer_seconds)
        self._buffer: Deque[TemporalFeature] = deque(maxlen=self._maxlen)
        # Incremental embedding sum for O(1) mean computation
        self._emb_sum: Optional[np.ndarray] = None
        self._boundary_times: List[float] = []

    # ------------------------------------------------------------------
    # Core buffer operations
    # ------------------------------------------------------------------

    def push(self, tf: TemporalFeature) -> None:
        """Append a TemporalFeature; evict oldest if at capacity."""
        if len(self._buffer) == self._maxlen and self._buffer:
            # Subtract evicted feature's embedding
            evicted = self._buffer[0]
            if self._emb_sum is not None:
                self._emb_sum -= evicted.multimodal_embedding

        self._buffer.append(tf)

        # Update incremental sum
        emb = tf.multimodal_embedding
        if self._emb_sum is None:
            self._emb_sum = emb.copy().astype(np.float64)
        else:
            self._emb_sum += emb.astype(np.float64)

    def record_boundary(self, timestamp_ms: float) -> None:
        """Record that a confirmed event boundary occurred at this timestamp."""
        self._boundary_times.append(timestamp_ms)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def mean_embedding(self) -> Optional[np.ndarray]:
        """
        Return the running mean of all embeddings in the buffer.
        Returns None if the buffer is empty.
        O(1) — uses incremental sum.
        """
        n = len(self._buffer)
        if n == 0 or self._emb_sum is None:
            return None
        return (self._emb_sum / n).astype(np.float32)

    def recent(self, n: int) -> List[TemporalFeature]:
        """Return the last n TemporalFeatures (or all if n > size)."""
        items = list(self._buffer)
        return items[-n:] if n < len(items) else items

    def recent_embeddings(self, n: int) -> List[np.ndarray]:
        """Return a list of the last n embedding vectors."""
        return [tf.multimodal_embedding for tf in self.recent(n)]

    def last_boundary_time(self) -> Optional[float]:
        """Return the timestamp_ms of the most recently confirmed boundary, or None."""
        return self._boundary_times[-1] if self._boundary_times else None

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)
