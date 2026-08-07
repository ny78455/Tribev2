"""
asvl/asvl/scheduler.py
Frame Scheduler — decides which frames to emit based on target FPS.

Logic: accumulator-based keep/drop.
A frame is emitted when its timestamp >= last_emitted_timestamp + 1000/target_fps.

Hard invariant (§5.8): never emit a frame with timestamp_ms <= previous emitted timestamp_ms.

Supports both sync (queue.Queue) and async (asyncio.Queue) output modes.
"""
import asyncio
import logging
import queue
from typing import Iterator, Optional

from .types import FramePacket

logger = logging.getLogger(__name__)


class FrameScheduler:
    """
    Accumulator-based frame scheduler.

    Given a stream of frames with per-frame target_fps, decides
    which frames to emit. Pushes kept frames onto an output queue.

    Modes:
        "sync"  → uses queue.Queue  (thread-safe blocking)
        "async" → uses asyncio.Queue (for async consumers)
    """

    def __init__(self, mode: str = "sync", maxsize: int = 256) -> None:
        """
        Args:
            mode: "sync" or "async".
            maxsize: Maximum queue depth.
        """
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be 'sync' or 'async', got '{mode}'")

        self.mode = mode
        self._last_emitted_ts: float = -1.0
        self._last_target_fps: float = 1.0

        if mode == "sync":
            self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
            self._async_queue: Optional[asyncio.Queue] = None
        else:
            self._queue = None
            self._async_queue = asyncio.Queue(maxsize=maxsize)

    # ------------------------------------------------------------------
    # Core scheduling logic
    # ------------------------------------------------------------------

    def should_emit(self, timestamp_ms: float, target_fps: float) -> bool:
        """
        Decide whether to emit this frame.

        Args:
            timestamp_ms: Frame timestamp in milliseconds.
            target_fps: Target sampling rate in frames per second.

        Returns:
            True if the frame should be emitted.
        """
        # Enforce hard monotonicity invariant
        if timestamp_ms <= self._last_emitted_ts:
            logger.debug(
                "Scheduler: skipping non-monotonic frame ts=%.1f <= last=%.1f",
                timestamp_ms,
                self._last_emitted_ts,
            )
            return False

        target_fps = max(target_fps, 0.01)  # guard against division by zero
        min_interval_ms = 1000.0 / target_fps

        # First frame always emitted
        if self._last_emitted_ts < 0:
            return True

        return (timestamp_ms - self._last_emitted_ts) >= min_interval_ms

    def emit(self, packet: FramePacket) -> bool:
        """
        Emit a FramePacket to the output queue (sync mode).

        Returns:
            True if successfully enqueued, False if queue is full.
        """
        if self.mode != "sync":
            raise RuntimeError("Call async_emit() in async mode.")

        try:
            self._queue.put_nowait(packet)
            self._last_emitted_ts = packet.timestamp_ms
            return True
        except queue.Full:
            logger.warning("Scheduler output queue full — dropping frame ts=%.1f", packet.timestamp_ms)
            return False

    async def async_emit(self, packet: FramePacket) -> bool:
        """
        Emit a FramePacket to the async output queue.

        Returns:
            True if successfully enqueued.
        """
        if self.mode != "async":
            raise RuntimeError("Call emit() in sync mode.")

        try:
            self._async_queue.put_nowait(packet)
            self._last_emitted_ts = packet.timestamp_ms
            return True
        except asyncio.QueueFull:
            logger.warning("Async scheduler queue full — dropping frame ts=%.1f", packet.timestamp_ms)
            return False

    def process(self, packet: FramePacket, target_fps: float) -> bool:
        """
        Convenience method: check + emit in one call (sync mode).

        Updates last_emitted_ts only if the frame is kept.

        Returns:
            True if the frame was emitted.
        """
        if self.should_emit(packet.timestamp_ms, target_fps):
            return self.emit(packet)
        return False

    # ------------------------------------------------------------------
    # Queue access
    # ------------------------------------------------------------------

    def get_queue(self) -> queue.Queue:
        """Return the sync output queue."""
        if self.mode != "sync":
            raise RuntimeError("No sync queue in async mode.")
        return self._queue

    def get_async_queue(self) -> asyncio.Queue:
        """Return the async output queue."""
        if self.mode != "async":
            raise RuntimeError("No async queue in sync mode.")
        return self._async_queue

    def drain(self) -> Iterator[FramePacket]:
        """Drain all currently queued packets (sync mode). Non-blocking."""
        if self.mode != "sync":
            raise RuntimeError("drain() is only for sync mode.")
        while not self._queue.empty():
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                break

    def reset(self) -> None:
        """Reset scheduler state (e.g. after seeking)."""
        self._last_emitted_ts = -1.0
