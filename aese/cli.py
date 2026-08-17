#!/usr/bin/env python3
"""
aese/cli.py
Command-line interface for AESE — Adaptive Event Segmentation Engine.

Usage:
    python cli.py --input out/manifest.jsonl --output events.jsonl
    python cli.py --input out/manifest.jsonl --output events.jsonl \\
                  --video ../comedy.mp4 --config config.default.yaml \\
                  --save-keyframes --verbose

Input modes:
  manifest-replay: --input points to a manifest.jsonl (no pixel data)
  live (future):   pipe ASVL FramePackets directly via --input -

Output format (one JSON per line, no embedding arrays):
  {"event_id": 0, "start_time_ms": 0, "end_time_ms": 26000, "duration_ms": 26000,
   "importance": 0.31, "confidence": 0.87, "summary": "...", "boundary_reason": "scene",
   "event_type": "Dialogue"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterator, Optional

import numpy as np

# Ensure aese package is importable when running cli.py from repo root
sys.path.insert(0, os.path.dirname(__file__))

from aese.config import load_config
from aese.pipeline import run
from aese.preflight import audit_manifest, print_report
from aese.types import AESEConfig, Event, FramePacket


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def _load_manifest(
    manifest_path: str,
    video_path: Optional[str] = None,
) -> Iterator[FramePacket]:
    """
    Replay a manifest.jsonl file as a stream of FramePackets.
    In manifest-replay mode, .image is None (no pixel data).
    If --video is provided, load the actual frame from the video at the given timestamp.
    """
    video_cap = None
    if video_path is not None:
        try:
            import cv2
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                logging.warning("AESE CLI: Could not open video %s — using black frames", video_path)
                video_cap = None
        except ImportError:
            logging.warning("AESE CLI: OpenCV not available — using black frames")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Load frame image if video is available
            image = None
            if video_cap is not None:
                ts_ms = data.get("timestamp_ms", 0.0)
                video_cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
                ret, frame_bgr = video_cap.read()
                if ret:
                    import cv2 as _cv2
                    image = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)

            yield FramePacket(
                frame_id=data.get("frame_id", 0),
                timestamp_ms=float(data.get("timestamp_ms", 0.0)),
                fps_used=float(data.get("fps_used", 1.0)),
                motion_score=float(data.get("motion_score", 0.0)),
                scene_change=bool(data.get("scene_change", False)),
                audio_energy=float(data.get("audio_energy", 0.0)),
                novelty_score=float(data.get("novelty_score", 0.0)),
                decision_reason=str(data.get("decision_reason", "")),
                subtitle_text=data.get("subtitle_text"),
                image=image,
            )

    if video_cap is not None:
        video_cap.release()


def _event_to_dict(event: Event, include_embedding: bool = False) -> dict:
    """Serialize an Event to a manifest-line dict."""
    d = {
        "event_id": event.event_id,
        "start_time_ms": round(event.start_time_ms, 1),
        "end_time_ms": round(event.end_time_ms, 1),
        "duration_ms": round(event.duration_ms, 1),
        "importance": round(float(event.importance), 4),
        "confidence": round(float(event.confidence), 4),
        "summary": event.summary,
        "boundary_reason": event.boundary_reason,
        "event_type": event.event_type,
        "location_label": event.location_label,
        # character_count_range: sorted unique per-second face *counts* — NOT entity IDs.
        # null means no image data was available for this event (manifest-replay without --video).
        # e.g. [0, 1, 2] means some seconds had 0 faces, some 1, some 2 — not 3 people identified.
        # See DECISIONS.md §4 and §14.
        "character_count_range": event.character_count_range,
        "max_characters_seen": event.max_characters_seen,
        "character_data_available": event.character_data_available,
    }
    if include_embedding and event.event_embedding is not None:
        # Emit embedding as a compact float list
        d["event_embedding"] = [round(float(v), 6) for v in event.event_embedding[:32]]
        d["event_embedding_dim"] = int(event.event_embedding.shape[0])
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aese",
        description="Adaptive Event Segmentation Engine — convert FramePackets to Events.",
    )
    parser.add_argument(
        "--input", required=True, metavar="MANIFEST",
        help="Path to manifest.jsonl (Module 1 output).",
    )
    parser.add_argument(
        "--output", required=True, metavar="EVENTS",
        help="Output path for events.jsonl.",
    )
    parser.add_argument(
        "--video", default=None, metavar="MP4",
        help="Optional path to the source video file. Enables real frame extraction "
             "for image-dependent adapters (embedding, scene label, character count).",
    )
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to YAML config file (default: built-in defaults).",
    )
    parser.add_argument(
        "--save-keyframes", action="store_true",
        help="Save keyframe embeddings in the output JSONL (first 32 dims only).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override boundary_threshold (0-1).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("aese.cli")

    # --- Validate inputs ---
    if not os.path.isfile(args.input):
        logger.error("Input manifest not found: %s", args.input)
        sys.exit(1)

    if args.video and not os.path.isfile(args.video):
        logger.warning("Video file not found: %s — running in manifest-only mode", args.video)
        args.video = None

    # --- Load config ---
    overrides = {}
    if args.threshold is not None:
        overrides["boundary_threshold"] = args.threshold

    try:
        config = load_config(yaml_path=args.config, overrides=overrides or None)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    if args.video is None:
        logger.warning(
            "\n" + "=" * 70 + "\n"
            "  RUNNING IN MANIFEST-REPLAY MODE WITHOUT --video\n"
            "  Character count, scene label, and embedding-based signals will be\n"
            "  UNAVAILABLE for this run. Event boundaries will rely only on\n"
            "  motion, audio, subtitle, and camera-cut (scene_change) signals.\n"
            "  Pass --video <path> for full-fidelity output.\n"
            + "=" * 70
        )

    logger.info(
        "AESE CLI: input=%s | video=%s | threshold=%.2f | output=%s",
        args.input, args.video or "none (manifest-replay mode)", config.boundary_threshold, args.output,
    )

    # --- Detector-mode banner (Fix 1) ---
    # Must print unconditionally to stdout — not just logger.warning — so it
    # cannot be filtered by log level settings or redirected output.
    from aese.adapters.fastvlm import _ensure_loaded
    from aese.adapters.character_stub import get_effective_detector_chain
    _ensure_loaded()  # trigger load attempt before reading the mode
    active_chain = get_effective_detector_chain()
    print("=" * 70)
    print(f"  CHARACTER DETECTION MODE: {active_chain}")
    if active_chain != "fastvlm":
        print(
            "  WARNING: FastVLM is NOT active. Character counts will use a\n"
            f"  weaker fallback detector ({active_chain}) with lower recall on\n"
            "  non-frontal faces, small faces, or difficult lighting.\n"
            "  Install torch, transformers>=4.52, and accelerate to enable\n"
            "  FastVLM-powered detection."
        )
    print("=" * 70)

    # --- Load manifest into memory for pre-flight audit (Fix 3) ---
    # Buffering into a list allows audit_manifest() to compute stats before
    # the pipeline consumes the stream. For typical movie manifests (~few
    # thousand packets at 1-2 fps) the memory overhead is negligible.
    packet_list = list(_load_manifest(args.input, video_path=args.video))

    # --- Signal-richness pre-flight report (Fix 3) ---
    report = audit_manifest(packet_list, detector_mode=active_chain)
    print_report(report)

    # --- Run pipeline ---
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    event_count = 0
    try:
        packet_stream = iter(packet_list)

        with open(args.output, "w", encoding="utf-8") as out_fh:
            for event in run(packet_stream, config):
                event_count += 1
                row = _event_to_dict(event, include_embedding=args.save_keyframes)
                out_fh.write(json.dumps(row) + "\n")
                logger.info(
                    "Event %d: [%.1f ms → %.1f ms] (%.1f s) %s | %s | conf=%.2f | reason=%s",
                    event.event_id,
                    event.start_time_ms,
                    event.end_time_ms,
                    event.duration_ms / 1000,
                    event.event_type,
                    event.summary,
                    event.confidence,
                    event.boundary_reason,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info(
        "AESE CLI done. %d events written to %s", event_count, args.output
    )


if __name__ == "__main__":
    main()
