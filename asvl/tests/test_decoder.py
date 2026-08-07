"""
tests/test_decoder.py
Acceptance tests for asvl.decoder (§5.1).

Uses a synthetically generated MP4 to avoid any external file dependency.
The synthetic video is created via PyAV (not cv2.VideoCapture).
"""
import math
import sys
import os
import tempfile

import av
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.decoder import decode_frames, probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(
    path: str,
    duration_s: float = 5.0,
    fps: int = 24,
    width: int = 320,
    height: int = 240,
) -> None:
    """
    Create a minimal synthetic H.264 mp4 for testing.
    Frames alternate between a red and blue solid color.
    """
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    total_frames = int(duration_s * fps)
    for i in range(total_frames):
        color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
        frame_data = np.full((height, width, 3), color, dtype=np.uint8)
        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        av_frame.pts = i
        av_frame.time_base = f"1/{fps}"
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProbe:
    def test_probe_returns_expected_keys(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=2.0, fps=24, width=320, height=240)

        info = probe(video_path)
        assert "fps" in info
        assert "duration" in info
        assert "resolution" in info
        assert "codec" in info
        assert "bitrate" in info

    def test_probe_fps_is_accurate(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=2.0, fps=24)
        info = probe(video_path)
        assert abs(info["fps"] - 24.0) < 1.0, f"Expected fps≈24, got {info['fps']}"

    def test_probe_duration_is_approximate(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=5.0, fps=24)
        info = probe(video_path)
        assert abs(info["duration"] - 5.0) < 1.0, f"Expected duration≈5s, got {info['duration']}"

    def test_probe_resolution(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=1.0, fps=24, width=320, height=240)
        info = probe(video_path)
        assert info["resolution"] == (320, 240)


class TestDecodeFrames:
    def test_timestamps_strictly_increasing(self, tmp_path):
        """Acceptance test §5.1: timestamps must be strictly increasing."""
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=5.0, fps=24)

        timestamps = [ts for _, ts in decode_frames(video_path)]
        assert len(timestamps) > 0, "No frames decoded"

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Non-monotonic timestamp at index {i}: "
                f"{timestamps[i]} <= {timestamps[i - 1]}"
            )

    def test_frame_count_within_tolerance(self, tmp_path):
        """Acceptance test §5.1: frame count within ±1 of duration * fps."""
        fps = 24
        duration = 5.0
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=duration, fps=fps)

        frames = list(decode_frames(video_path))
        expected = int(duration * fps)
        actual = len(frames)
        assert abs(actual - expected) <= 1, (
            f"Frame count {actual} deviates from expected {expected} by more than ±1"
        )

    def test_frames_are_rgb(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=1.0, fps=24, width=320, height=240)

        for frame, _ in decode_frames(video_path):
            assert frame.ndim == 3, "Frame is not 3D"
            assert frame.shape[2] == 3, "Frame does not have 3 channels"
            assert frame.dtype == np.uint8, "Frame dtype is not uint8"
            break  # one frame is enough

    def test_decode_does_not_load_full_file(self, tmp_path):
        """Ensure decode_frames is a generator (lazy), not a list."""
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, duration_s=5.0, fps=24)

        gen = decode_frames(video_path)
        import inspect
        assert inspect.isgenerator(gen), "decode_frames must return a generator"
