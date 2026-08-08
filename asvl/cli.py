#!/usr/bin/env python3
"""
asvl/cli.py
Command-line interface for ASVL — Adaptive Streaming Video Loader.

Usage:
    python cli.py --input movie.mp4 --output out/
    python cli.py --input movie.mp4 --output out/ --config config.default.yaml \\
                  --subtitles movie.srt --mode sync --save-frames --save-manifest
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

# Ensure asvl package is importable when running cli.py from repo root
sys.path.insert(0, os.path.dirname(__file__))

from asvl.config import load_config
from asvl.pipeline import run
from asvl.types import FramePacket


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def _packet_to_dict(packet: FramePacket) -> dict:
    """Serialize a FramePacket to a manifest-line dict (no raw image)."""
    return {
        "frame_id": packet.frame_id,
        "timestamp_ms": round(packet.timestamp_ms, 3),
        "fps_used": packet.fps_used,
        "motion_score": round(packet.motion_score, 4),
        "scene_change": packet.scene_change,
        "audio_energy": round(packet.audio_energy, 4),
        "novelty_score": round(packet.novelty_score, 4),
        "decision_reason": packet.decision_reason,
        "subtitle_text": packet.subtitle_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="asvl",
        description="Adaptive Streaming Video Loader — emit only important frames.",
    )
    parser.add_argument("--input", required=True, metavar="VIDEO", help="Path to input video file.")
    parser.add_argument("--output", required=True, metavar="DIR", help="Output directory.")
    parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Path to YAML config file (default: built-in defaults).",
    )
    parser.add_argument("--subtitles", default=None, metavar="SRT/VTT", help="Optional subtitle file.")
    parser.add_argument(
        "--mode",
        default="sync",
        choices=["sync", "async"],
        help="Scheduler mode (default: sync).",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Dump kept frames as JPEGs into <output>/frames/.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Suppress writing <output>/manifest.jsonl (manifest is written by default).",
    )
    # Legacy alias kept for backward compatibility
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden — manifest is now written by default
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    # Config override flags
    parser.add_argument("--min-fps", type=float, default=None, help="Override minimum_fps.")
    parser.add_argument("--max-fps", type=float, default=None, help="Override maximum_fps.")
    parser.add_argument("--adaptive", type=lambda x: x.lower() == "true", default=None)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger("asvl.cli")

    # --- Load config ---
    overrides = {}
    if args.min_fps is not None:
        overrides["minimum_fps"] = args.min_fps
    if args.max_fps is not None:
        overrides["maximum_fps"] = args.max_fps
    if args.adaptive is not None:
        overrides["adaptive"] = args.adaptive

    try:
        config = load_config(yaml_path=args.config, overrides=overrides or None)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    # --- Validate input ---
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    # --- Prepare output directories ---
    os.makedirs(args.output, exist_ok=True)
    frames_dir = os.path.join(args.output, "frames")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    manifest_path = os.path.join(args.output, "manifest.jsonl")
    manifest_file = None
    # Write manifest by default; suppress with --no-manifest.
    write_manifest = not args.no_manifest
    if write_manifest:
        manifest_file = open(manifest_path, "w", encoding="utf-8")
        logger.info("Writing manifest to: %s", manifest_path)

    # --- Run pipeline ---
    kept_count = 0
    try:
        for packet in run(
            video_path=args.input,
            config=config,
            subtitle_path=args.subtitles,
            scheduler_mode=args.mode,
        ):
            kept_count += 1

            # Save frame image
            if args.save_frames:
                import cv2
                frame_path = os.path.join(frames_dir, f"frame_{packet.frame_id:08d}.jpg")
                # Pipeline yields RGB; cv2.imwrite expects BGR
                bgr = cv2.cvtColor(packet.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, bgr)

            # Write manifest line
            if manifest_file is not None:
                line = json.dumps(_packet_to_dict(packet))
                manifest_file.write(line + "\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        if manifest_file is not None:
            manifest_file.close()

    logger.info("Done. Kept %d frames. Output: %s", kept_count, args.output)


if __name__ == "__main__":
    main()
