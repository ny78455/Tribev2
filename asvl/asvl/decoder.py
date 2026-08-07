"""
asvl/asvl/decoder.py
PyAV-based video decoder — streaming, never reads the whole file into memory.

Uses precise PTS (Presentation Timestamps) for accurate frame timing.
Integrates failure callbacks from failure_handling.py.
"""
import logging
from typing import Iterator, Tuple

import av
import numpy as np

from .failure_handling import on_corrupted_frame, on_decode_failure, on_missing_timestamp

logger = logging.getLogger(__name__)


def probe(path: str) -> dict:
    """
    Probe a video file and return metadata.

    Args:
        path: Path to video file (.mp4, .mkv, .avi, .mov, .webm).

    Returns:
        dict with keys: fps, duration, resolution, codec, bitrate.
    """
    container = av.open(path)
    stream = container.streams.video[0]

    fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    resolution = (stream.width, stream.height)
    codec = stream.codec_context.name
    bitrate = int(container.bit_rate) if container.bit_rate else 0

    container.close()

    return {
        "fps": fps,
        "duration": duration,
        "resolution": resolution,
        "codec": codec,
        "bitrate": bitrate,
    }


def decode_frames(path: str) -> Iterator[Tuple[np.ndarray, float]]:
    """
    Stream-decode a video file and yield (frame_rgb, timestamp_ms) pairs.

    - Uses PyAV for accurate PTS — never cv2.VideoCapture.
    - Streams frames; never loads the whole file into memory.
    - Timestamps are guaranteed to be strictly increasing (or estimated if missing).
    - Corrupted/malformed frames are skipped with a WARNING log.
    - Decode failures trigger a seek-to-keyframe recovery attempt.

    Args:
        path: Path to video file.

    Yields:
        (frame: np.ndarray HxWx3 uint8 RGB, timestamp_ms: float)
    """
    container = av.open(path)
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"  # multithreaded decoding

    # Get fps for missing-timestamp estimation
    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
    time_base = float(video_stream.time_base)

    frame_index = 0
    prev_ts_ms = -1.0

    try:
        for packet in container.demux(video_stream):
            try:
                frames = packet.decode()
            except av.AVError as exc:
                logger.warning("Packet decode error at frame %d: %s", frame_index, exc)
                recovered = on_decode_failure(container, video_stream)
                if not recovered:
                    break
                continue

            for av_frame in frames:
                # --- Timestamp resolution ---
                if av_frame.pts is not None:
                    timestamp_ms = float(av_frame.pts * time_base * 1000.0)
                else:
                    timestamp_ms = on_missing_timestamp(
                        prev_ts_ms if prev_ts_ms >= 0 else 0.0, fps
                    )

                # Enforce strict monotonicity — drop frames with duplicate/out-of-order PTS
                if timestamp_ms <= prev_ts_ms:
                    logger.debug(
                        "Non-monotonic timestamp %.1f <= %.1f at frame %d; skipping.",
                        timestamp_ms,
                        prev_ts_ms,
                        frame_index,
                    )
                    frame_index += 1
                    continue

                # --- Frame conversion ---
                try:
                    frame_rgb = av_frame.to_ndarray(format="rgb24")
                except Exception as exc:
                    on_corrupted_frame(frame_index, timestamp_ms)
                    logger.debug("Frame conversion error: %s", exc)
                    frame_index += 1
                    continue

                # Basic sanity check for corrupted/zero frames
                if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
                    on_corrupted_frame(frame_index, timestamp_ms)
                    frame_index += 1
                    continue

                prev_ts_ms = timestamp_ms
                frame_index += 1
                yield frame_rgb, timestamp_ms

    finally:
        container.close()
