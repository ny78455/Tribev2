"""
tests/test_scheduler.py
Acceptance tests for asvl.scheduler (§5.8).

Feeds monotonic timestamps at 24fps with alternating target_fps of 0.5 and 10.
Asserts:
  - Output timestamps strictly increasing.
  - Output matches expected sparsity (0.5fps → ~2s intervals, 10fps → ~0.1s intervals).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.scheduler import FrameScheduler
from asvl.types import FramePacket


def _make_packet(frame_id: int, timestamp_ms: float, fps: float = 5.0) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        fps_used=fps,
        motion_score=0.0,
        scene_change=False,
        audio_energy=0.0,
        novelty_score=0.0,
        decision_reason="test",
    )


class TestSchedulerMonotonicity:
    def test_output_timestamps_strictly_increasing(self):
        """Acceptance test §5.8: output timestamps must be strictly increasing."""
        scheduler = FrameScheduler(mode="sync")
        native_fps = 24
        duration_ms = 10_000  # 10 seconds

        timestamps = [i * (1000.0 / native_fps) for i in range(int(native_fps * duration_ms / 1000))]

        for i, ts in enumerate(timestamps):
            # Alternate between sparse (0.5fps) and dense (10fps)
            target_fps = 0.5 if (i // 24) % 2 == 0 else 10.0
            packet = _make_packet(i, ts, target_fps)
            scheduler.process(packet, target_fps)

        kept = list(scheduler.drain())

        # Check strict monotonicity
        for j in range(1, len(kept)):
            assert kept[j].timestamp_ms > kept[j - 1].timestamp_ms, (
                f"Non-monotonic: {kept[j].timestamp_ms} <= {kept[j - 1].timestamp_ms} at index {j}"
            )

    def test_no_emission_for_non_monotonic_input(self):
        """Frames with timestamp <= last emitted must not be emitted."""
        scheduler = FrameScheduler(mode="sync")
        packet_a = _make_packet(0, 1000.0, 10.0)
        packet_b = _make_packet(1, 500.0, 10.0)  # earlier timestamp

        scheduler.process(packet_a, 10.0)
        scheduler.process(packet_b, 10.0)  # should be dropped

        kept = list(scheduler.drain())
        assert len(kept) == 1
        assert kept[0].timestamp_ms == 1000.0


class TestSchedulerSparsity:
    def test_sparse_mode_limits_output(self):
        """At 0.5fps, frames should be ~2000ms apart."""
        scheduler = FrameScheduler(mode="sync")
        native_fps = 24
        total_ms = 10_000

        for i in range(int(native_fps * total_ms / 1000)):
            ts = i * (1000.0 / native_fps)
            packet = _make_packet(i, ts)
            scheduler.process(packet, 0.5)

        kept = list(scheduler.drain())
        # At 0.5fps over 10s, expect ~5 frames (10 * 0.5 = 5)
        # Allow generous range due to accumulator boundary effects
        assert 3 <= len(kept) <= 7, f"Expected ~5 frames at 0.5fps, got {len(kept)}"

    def test_dense_mode_passes_many_frames(self):
        """At 10fps from 24fps source, most frames should pass."""
        scheduler = FrameScheduler(mode="sync")
        native_fps = 24
        total_ms = 5_000

        for i in range(int(native_fps * total_ms / 1000)):
            ts = i * (1000.0 / native_fps)
            packet = _make_packet(i, ts)
            scheduler.process(packet, 10.0)

        kept = list(scheduler.drain())
        # At 10fps from 24fps for 5s → expect ~50 frames
        assert 40 <= len(kept) <= 55, f"Expected ~50 frames at 10fps, got {len(kept)}"


class TestSchedulerFirstFrame:
    def test_first_frame_always_emitted(self):
        scheduler = FrameScheduler(mode="sync")
        packet = _make_packet(0, 0.0, 0.5)
        scheduler.process(packet, 0.5)
        kept = list(scheduler.drain())
        assert len(kept) == 1


class TestSchedulerReset:
    def test_reset_clears_state(self):
        scheduler = FrameScheduler(mode="sync")
        packet = _make_packet(0, 5000.0, 10.0)
        scheduler.process(packet, 10.0)
        list(scheduler.drain())

        scheduler.reset()
        # After reset, a frame at 0ms should be emitted again
        packet2 = _make_packet(1, 0.0, 10.0)
        scheduler.process(packet2, 10.0)
        kept = list(scheduler.drain())
        # After reset, first frame should always be emitted
        assert len(kept) == 1
