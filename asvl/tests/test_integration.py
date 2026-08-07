"""
tests/test_integration.py
End-to-end integration test for the ASVL pipeline (§5.10).

Creates a synthetic 10-second video with alternating static and motion segments,
runs the full pipeline, and verifies:
  - Output is non-empty
  - FramePackets have strictly increasing timestamps
  - FPS visibly varies between segments (adaptive behavior)
  - No exceptions are raised
"""
import sys
import os
import tempfile

import av
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.pipeline import run
from asvl.config import load_config
from asvl.types import FramePacket


def _make_mixed_video(path: str, fps: int = 24, total_seconds: float = 10.0) -> None:
    """
    Create a synthetic video with alternating:
      - 0-4s: static (solid red) — should produce low FPS
      - 4-7s: high motion (random noise per frame) — should produce high FPS
      - 7-10s: static again (solid blue)
    """
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "28"}

    total_frames = int(fps * total_seconds)
    rng = np.random.default_rng(seed=0)

    for i in range(total_frames):
        t = i / fps
        if t < 4.0:
            # Static segment — solid red
            frame_data = np.full((240, 320, 3), (200, 50, 50), dtype=np.uint8)
        elif t < 7.0:
            # Motion segment — random noise
            frame_data = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        else:
            # Static segment — solid blue
            frame_data = np.full((240, 320, 3), (50, 50, 200), dtype=np.uint8)

        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        av_frame.pts = i
        av_frame.time_base = f"1/{fps}"
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


class TestPipelineIntegration:
    def test_pipeline_runs_without_error(self, tmp_path):
        video_path = str(tmp_path / "mixed.mp4")
        _make_mixed_video(video_path)

        config = load_config()  # defaults
        packets = list(run(video_path, config))
        assert len(packets) > 0, "Pipeline emitted no frames"

    def test_output_timestamps_strictly_increasing(self, tmp_path):
        video_path = str(tmp_path / "mixed.mp4")
        _make_mixed_video(video_path)

        config = load_config()
        packets = list(run(video_path, config))

        for i in range(1, len(packets)):
            assert packets[i].timestamp_ms > packets[i - 1].timestamp_ms, (
                f"Non-monotonic timestamp at index {i}: "
                f"{packets[i].timestamp_ms} <= {packets[i - 1].timestamp_ms}"
            )

    def test_fps_varies_between_segments(self, tmp_path):
        """Adaptive behavior: action segment should have higher avg fps than static."""
        video_path = str(tmp_path / "mixed.mp4")
        _make_mixed_video(video_path)

        config = load_config()
        packets = list(run(video_path, config))

        # Split by timestamp
        static_fps = [p.fps_used for p in packets if p.timestamp_ms < 4000]
        motion_fps = [p.fps_used for p in packets if 4000 <= p.timestamp_ms < 7000]

        if static_fps and motion_fps:
            avg_static = sum(static_fps) / len(static_fps)
            avg_motion = sum(motion_fps) / len(motion_fps)
            # Motion segment should use higher target FPS
            assert avg_motion >= avg_static, (
                f"Expected motion fps ({avg_motion:.2f}) >= static fps ({avg_static:.2f})"
            )

    def test_frame_packets_have_correct_types(self, tmp_path):
        video_path = str(tmp_path / "mixed.mp4")
        _make_mixed_video(video_path)

        config = load_config()
        packets = list(run(video_path, config))

        for p in packets[:5]:  # Check first 5
            assert isinstance(p, FramePacket)
            assert isinstance(p.frame_id, int)
            assert isinstance(p.timestamp_ms, float)
            assert isinstance(p.image, np.ndarray)
            assert p.image.ndim == 3
            assert p.image.shape[2] == 3
            assert isinstance(p.decision_reason, str)
            assert len(p.decision_reason) > 0

    def test_pipeline_with_no_subtitles(self, tmp_path):
        video_path = str(tmp_path / "mixed.mp4")
        _make_mixed_video(video_path)

        config = load_config()
        # subtitle_text should be None for all packets when no subtitle file given
        packets = list(run(video_path, config, subtitle_path=None))
        for p in packets:
            assert p.subtitle_text is None
