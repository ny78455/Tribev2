"""
asvl/asvl/failure_handling.py
Recovery logic for decode failures, corrupted frames, and missing timestamps.
Called from decoder.py via callbacks — never silently swallowed.
"""
import logging

logger = logging.getLogger(__name__)


def on_decode_failure(container, stream) -> bool:
    """
    Called when a packet decode fails.
    Attempts to seek to the nearest prior keyframe and resume.
    Returns True if recovery is possible, False if the stream is unrecoverable.
    """
    try:
        # Seek to current position — PyAV will align to nearest keyframe
        current_pts = stream.codec_context.pts if hasattr(stream, "codec_context") else 0
        container.seek(max(0, current_pts - 1), backward=True, any_frame=False)
        logger.warning("Decode failure: sought to prior keyframe for recovery.")
        return True
    except Exception as exc:
        logger.error("Decode failure: unable to recover from stream error: %s", exc)
        return False


def on_corrupted_frame(frame_index: int, timestamp_ms: float) -> None:
    """
    Called when a decoded frame is detected as corrupted (e.g. all-zero, wrong shape).
    Logs a WARNING and signals the caller to skip this frame.
    """
    logger.warning(
        "Corrupted frame detected at index=%d timestamp_ms=%.1f — skipping.",
        frame_index,
        timestamp_ms,
    )


def on_missing_timestamp(prev_ts: float, fps: float) -> float:
    """
    Called when a frame has no valid PTS.
    Returns an estimated timestamp: prev_ts + 1000/fps (ms).
    """
    estimated = prev_ts + (1000.0 / fps if fps > 0 else 40.0)
    logger.warning(
        "Missing timestamp: estimating %.1f ms from prev_ts=%.1f fps=%.2f",
        estimated,
        prev_ts,
        fps,
    )
    return estimated
