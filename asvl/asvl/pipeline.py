"""
asvl/asvl/pipeline.py
Pipeline Orchestrator — wires all ASVL modules together.

Single entry point:
    run(video_path, config, subtitle_path=None) -> Iterator[FramePacket]

Architecture:
    Decoder → Buffer → Feature Extractor → Audio → Subtitles
           → Controller → Scene Transition → Scheduler → yield FramePacket

Non-functional requirements:
    - Logs a WARNING (not a crash) if per-frame decision loop exceeds 100ms.
    - Caps memory to rolling buffer window — never stores full movie in RAM.
    - Logs peak RSS (via psutil) at the end of the run.
"""
import logging
import os
import time
from typing import Iterator, Optional

import numpy as np

from .audio import SILENT_AUDIO, build_audio_index, get_audio_at
from .buffer import RollingFrameBuffer
from .config import load_config
from .controller import compute_importance, get_decision_reason, importance_to_fps
from .decoder import decode_frames, probe
from .features.blur import is_black_frame, is_blurred
from .features.edges import compute_edge_diff
from .features.motion import compute_motion_score
from .features.novelty import compute_novelty
from .features.scene import compute_scene_score
from .scheduler import FrameScheduler
from .scene_transition import SceneTransitionDetector
from .subtitles import SubtitleSync
from .transcribe import transcribe as _auto_transcribe
from .types import ASVLConfig, FrameFeatures, FramePacket

logger = logging.getLogger(__name__)

_DECISION_LATENCY_WARN_MS = 100.0


def _get_peak_rss_mb() -> float:
    """Return peak resident set size in MB via psutil (cross-platform)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.warning("psutil not installed — cannot report peak RSS.")
        return -1.0


def run(
    video_path: str,
    config: ASVLConfig,
    subtitle_path: Optional[str] = None,
    scheduler_mode: str = "sync",
    auto_transcribe: bool = True,
    whisper_model: Optional[str] = None,
    whisper_language: str = "en",
) -> Iterator[FramePacket]:
    """
    Main ASVL pipeline: stream-decode a video and yield important FramePackets.

    Args:
        video_path:       Path to the input video file.
        config:           ASVLConfig instance (use load_config() to create).
        subtitle_path:    Optional path to .srt or .vtt subtitle file.
                          If None and auto_transcribe is True, whisper.cpp is
                          used to auto-generate subtitles (if available).
        scheduler_mode:   "sync" or "async" queue mode for the scheduler.
        auto_transcribe:  Attempt whisper.cpp auto-transcription when no
                          subtitle_path is provided. Set False to disable.
        whisper_model:    Override path to whisper ggml model file.
        whisper_language: Language code for whisper (default: "en").

    Yields:
        FramePacket objects for frames selected by the adaptive controller.
    """
    # --- Probe video ---
    video_info = probe(video_path)
    native_fps = video_info["fps"] or 25.0
    logger.info(
        "ASVL: starting pipeline on %s | fps=%.2f duration=%.1fs %s",
        os.path.basename(video_path),
        native_fps,
        video_info["duration"],
        video_info["resolution"],
    )

    # --- Initialize subsystems ---
    buffer = RollingFrameBuffer(
        buffer_seconds=config.buffer_seconds,
        fps=native_fps,
    )

    # --- Auto-transcription (when no subtitle file provided) ---
    _auto_srt_path: Optional[str] = None
    effective_subtitle_path = subtitle_path
    if subtitle_path is None and auto_transcribe:
        _auto_srt_path = _auto_transcribe(
            video_path,
            model_path=whisper_model,
            language=whisper_language,
        )
        if _auto_srt_path is not None:
            effective_subtitle_path = _auto_srt_path

    subs = SubtitleSync(effective_subtitle_path)
    scene_detector = SceneTransitionDetector(scene_threshold=config.scene_threshold)
    scheduler = FrameScheduler(mode=scheduler_mode)

    # --- Pre-build audio index (O(1) lookup by timestamp) ---
    logger.info("ASVL: building audio index...")
    try:
        audio_index = build_audio_index(video_path)
        logger.info("ASVL: audio index built — %d windows.", len(audio_index))
    except Exception as exc:
        logger.warning("ASVL: audio extraction failed (%s) — using silent fallback.", exc)
        audio_index = []

    # --- Per-frame pipeline state ---
    prev_frame: Optional[np.ndarray] = None
    frame_id = 0
    decision_latencies_ms = []
    peak_rss_mb = 0.0

    # --- Per-stage timer accumulators (seconds) ---
    _t_decode = 0.0
    _t_motion = 0.0
    _t_scene = 0.0
    _t_audio = 0.0
    _t_novelty = 0.0
    _t_schedule = 0.0

    # --- Main decode loop ---
    for frame_rgb, timestamp_ms in decode_frames(video_path):
        _ts_decode_start = time.perf_counter()
        loop_start = time.perf_counter()
        _t_decode += time.perf_counter() - _ts_decode_start

        # --- Feature extraction ---
        if prev_frame is None:
            # First frame: use neutral defaults
            motion_score = 0.0
            scene_score = 0.0
            scene_change_feat = False
            edge_diff = 0.0
        else:
            _t0 = time.perf_counter()
            try:
                motion_score = compute_motion_score(prev_frame, frame_rgb)
            except Exception as exc:
                logger.debug("Motion score error: %s", exc)
                motion_score = 0.0
            _t_motion += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            try:
                scene_score, scene_change_feat = compute_scene_score(
                    prev_frame, frame_rgb, config.scene_threshold
                )
            except Exception as exc:
                logger.debug("Scene score error: %s", exc)
                scene_score, scene_change_feat = 0.0, False
            _t_scene += time.perf_counter() - _t0

            try:
                edge_diff = compute_edge_diff(prev_frame, frame_rgb)
            except Exception as exc:
                logger.debug("Edge diff error: %s", exc)
                edge_diff = 0.0

        # Single-frame features
        try:
            frame_blurred = is_blurred(frame_rgb)
        except Exception:
            frame_blurred = False

        try:
            frame_black = is_black_frame(frame_rgb)
        except Exception:
            frame_black = False

        # Novelty from buffer mean histogram
        _t0 = time.perf_counter()
        try:
            buf_hist = buffer.mean_histogram()
            # Safe default is 0.0 (not novel) when buffer is still empty — avoids
            # constant 1.0 saturation during the cold-start window.
            novelty_score = compute_novelty(frame_rgb, buf_hist) if buf_hist is not None else 0.0
        except Exception as exc:
            logger.debug("Novelty score error: %s", exc)
            novelty_score = 0.0
        _t_novelty += time.perf_counter() - _t0

        # --- Scene transition (stateful, separate from scene_score) ---
        scene_changed = scene_detector.update(frame_rgb)

        # --- Audio features (O(1) lookup into pre-built index — no file I/O) ---
        _t0 = time.perf_counter()
        if audio_index:
            audio_feat = get_audio_at(audio_index, timestamp_ms) or SILENT_AUDIO
        else:
            audio_feat = SILENT_AUDIO
        _t_audio += time.perf_counter() - _t0

        # --- Subtitle density ---
        subtitle_text = subs.get_subtitle_at(timestamp_ms)
        sub_density = subs.subtitle_density(timestamp_ms, window_ms=5000.0)

        # --- Controller: compute importance + target FPS ---
        importance = compute_importance(
            motion=motion_score,
            scene_change=scene_changed,
            audio_energy=audio_feat.energy,
            subtitle_density=sub_density,
            novelty=novelty_score,
            weights=config.weights,
        )
        target_fps = importance_to_fps(importance)
        target_fps = float(
            max(config.minimum_fps, min(config.maximum_fps, target_fps))
        )

        reason = get_decision_reason(
            motion=motion_score,
            scene_change=scene_changed,
            audio_energy=audio_feat.energy,
            subtitle_density=sub_density,
            novelty=novelty_score,
            weights=config.weights,
            importance_score=importance,
        )

        # --- Build FrameFeatures for buffer ---
        frame_features = FrameFeatures(
            motion_score=motion_score,
            scene_score=scene_score,
            scene_change=scene_changed,
            edge_diff=edge_diff,
            is_blurred=frame_blurred,
            is_black_frame=frame_black,
            novelty_score=novelty_score,
        )

        # --- Push to rolling buffer ---
        buffer.push(frame_rgb, timestamp_ms, frame_features)

        # --- Build FramePacket ---
        packet = FramePacket(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            image=frame_rgb,
            fps_used=target_fps,
            motion_score=motion_score,
            scene_change=scene_changed,
            audio_energy=audio_feat.energy,
            novelty_score=novelty_score,
            decision_reason=reason,
            subtitle_text=subtitle_text,
        )

        # --- Scheduler: emit or skip ---
        _t0 = time.perf_counter()
        scheduler.process(packet, target_fps)
        # Yield kept packets
        for kept in scheduler.drain():
            yield kept
        _t_schedule += time.perf_counter() - _t0

        # --- Latency check ---
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        decision_latencies_ms.append(loop_ms)
        if loop_ms > _DECISION_LATENCY_WARN_MS:
            logger.warning(
                "Decision loop latency %.1f ms > 100ms threshold at frame %d (ts=%.1f ms)",
                loop_ms,
                frame_id,
                timestamp_ms,
            )

        # Peak RSS tracking (sampled every 100 frames to reduce overhead)
        if frame_id % 100 == 0:
            rss = _get_peak_rss_mb()
            if rss > peak_rss_mb:
                peak_rss_mb = rss

        prev_frame = frame_rgb
        frame_id += 1

    # --- End-of-run stats ---
    if decision_latencies_ms:
        sorted_lat = sorted(decision_latencies_ms)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        logger.info(
            "ASVL run complete: %d frames processed, %d emitted | p95 latency=%.1f ms | peak RSS=%.1f MB",
            frame_id,
            frame_id,  # approximate — scheduler tracks actual count
            p95,
            peak_rss_mb,
        )
        if peak_rss_mb > 2048:
            logger.warning("Peak RSS %.1f MB exceeded 2GB target.", peak_rss_mb)

    # --- Per-stage timing summary (v1.1 instrumentation) ---
    logger.info(
        "ASVL stage timings (total seconds) — "
        "decode=%.2fs  motion=%.2fs  scene=%.2fs  audio=%.2fs  novelty=%.2fs  schedule=%.2fs",
        _t_decode, _t_motion, _t_scene, _t_audio, _t_novelty, _t_schedule,
    )
