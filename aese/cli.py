#!/usr/bin/env python3
"""
aese/cli.py
Command-line interface for AESE — Adaptive Event Segmentation Engine.

Usage:
    python cli.py --input out/manifest.jsonl --output events.jsonl
    python cli.py --input out/manifest.jsonl --output events.jsonl \\
                  --video ../comedy.mp4 --format human \\
                  --subtitles ../comedy.srt \\
                  --character-references refs/john.jpg=John refs/sarah.jpg=Sarah

Input modes:
  manifest-replay: --input points to a manifest.jsonl (no pixel data)
  live (future):   pipe ASVL FramePackets directly via --input -

Output formats:
  jsonl (default): one JSON per line, machine-readable canonical output.
  human:           human-readable event log (.txt sidecar alongside JSONL).
                   JSONL is always written; --format human adds the .txt on top.

Output format (one JSON per line, no embedding arrays):
  {"event_id": 0, "start_time_ms": 0, "end_time_ms": 26000, "duration_ms": 26000,
   "importance": 0.31, "confidence": 0.87, "summary": "...", "boundary_reason": "scene",
   "event_type": "Dialogue", "character_labels": ["Person A"], "location_label": "office"}
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
from aese.render import render_event_log
from aese.types import AESEConfig, Event, FramePacket
from aese.adapters.character_naming import CharacterNameBinder, apply_resolved_names


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
        # character_labels: anonymous-by-default consistent IDs (Fix 4) or real names
        # supplied via --character-references (Fix 5). Empty list in manifest-replay
        # mode without --video. Never guessed real names of unlabeled people.
        "character_labels": getattr(event, "character_labels", []),
        # character_count_range: sorted unique per-second face *counts* -- NOT entity IDs.
        # null means no image data was available for this event (manifest-replay without --video).
        # e.g. [0, 1, 2] means some seconds had 0 faces, some 1, some 2 -- not 3 people identified.
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
        "--format", choices=["jsonl", "human"], default="jsonl",
        help="Output format. 'jsonl' (default): machine-readable JSONL. "
             "'human': also writes a human-readable .txt sidecar alongside the JSONL.",
    )
    parser.add_argument(
        "--subtitles", default=None, metavar="SRT",
        help="Optional path to a .srt subtitle file. When supplied, dialogue text "
             "from subtitles is injected into the VLM summary prompt for richer narrative output.",
    )
    parser.add_argument(
        "--character-references", nargs="*", default=[], metavar="PATH=NAME",
        help="Optional named reference photos in 'path=name' format. "
             "Example: refs/john.jpg=John refs/sarah.jpg=Sarah. "
             "Only faces matching a supplied reference within threshold are assigned "
             "real names; all others remain anonymous (Person A, Person B, ...).",
    )
    parser.add_argument(
        "--vlm",
        choices=["fastvlm", "gemma4", "yunet"],
        default="fastvlm",
        metavar="BACKEND",
        help="VLM backend to use for scene labeling and event captioning. "
             "Choices: fastvlm (apple/FastVLM-0.5B, fast, low VRAM), "
             "gemma4 (google/gemma-4-E2B-it, high quality, needs >=24 GB VRAM), "
             "yunet (no generative VLM; character detection via OpenCV YuNet only). "
             "Default: fastvlm.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger("aese.cli")

    # --- Apply VLM backend selection ---
    from aese.adapters.vlm_router import set_backend as _set_vlm_backend
    _set_vlm_backend(args.vlm)

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

    # --- Detector-mode banner ---
    # Must print unconditionally to stdout -- not just logger.warning -- so it
    # cannot be filtered by log level settings or redirected output.
    from aese.adapters.vlm_router import get_active_detector_mode as _vlm_detector_mode
    from aese.adapters.vlm_router import vlm_available as _vlm_available_fn
    from aese.adapters.character_stub import get_effective_detector_chain
    from aese.adapters.scene_label import _clip_available as _scene_clip_available
    _vlm_available_fn()  # trigger lazy load attempt for the selected backend
    active_chain = get_effective_detector_chain()
    vlm_mode = _vlm_detector_mode()
    print("=" * 70)
    print(f"  VLM BACKEND: {args.vlm.upper()}  (active mode: {vlm_mode})")
    print(f"  CHARACTER DETECTION MODE: {active_chain}")
    if active_chain not in ("fastvlm", "yunet", "dnn", "opencv_haar_frontal+profile", "opencv_haar_frontal"):
        print(
            "  WARNING: No face detector is active. Character counts will be 0.\n"
            "  Install torch + transformers>=4.52 for fastvlm, or ensure\n"
            "  models/yunet/face_detection_yunet_2023mar.onnx exists for yunet."
        )
    scene_mode = "CLIP" if _scene_clip_available() else 'unavailable (labels will be "unknown" or heuristic)'
    print(f"  SCENE CLASSIFICATION MODE: {scene_mode}")
    print("=" * 70)

    # --- Load manifest into memory for pre-flight audit (Fix 3) ---
    # Buffering into a list allows audit_manifest() to compute stats before
    # the pipeline consumes the stream. For typical movie manifests (~few
    # thousand packets at 1-2 fps) the memory overhead is negligible.
    packet_list = list(_load_manifest(args.input, video_path=args.video))

    # --- Signal-richness pre-flight report (Fix 3) ---
    report = audit_manifest(packet_list, detector_mode=active_chain)
    print_report(report)

    # --- Parse --character-references ---
    reference_paths = {}  # {path: name}
    for ref in getattr(args, "character_references", []):
        if "=" in ref:
            path, name = ref.split("=", 1)
            reference_paths[path.strip()] = name.strip()
        else:
            logger.warning("Ignoring malformed --character-references entry (expected path=Name): %s", ref)

    # --- Run pipeline ---
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    all_events = []
    event_count = 0
    binder = CharacterNameBinder()  # accumulates vocative name evidence across the clip
    try:
        packet_stream = iter(packet_list)

        with open(args.output, "w", encoding="utf-8") as out_fh:
            for event in run(packet_stream, config, binder=binder):
                event_count += 1
                all_events.append(event)
                row = _event_to_dict(event, include_embedding=args.save_keyframes)
                out_fh.write(json.dumps(row) + "\n")
                logger.info(
                    "Event %d: [%.1f ms -> %.1f ms] (%.1f s) %s | %s | conf=%.2f | reason=%s",
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

    # --- Retroactive name relabeling (batch display-time pass) ---
    # apply_resolved_names() updates character_labels and summary text for ALL
    # events using whatever name evidence accumulated during the full clip run.
    # This is intentionally retroactive: an event at t=2s gets the name
    # first evidenced at t=15s. The online detection logic is never affected.
    # See DECISIONS.md for the online/batch boundary.
    apply_resolved_names(all_events, binder)
    if binder.resolved_names():
        # Re-write the output file with the updated labels
        logger.info("AESE CLI: re-writing %s with resolved character names", args.output)
        try:
            with open(args.output, "w", encoding="utf-8") as out_fh:
                for event in all_events:
                    row = _event_to_dict(event, include_embedding=args.save_keyframes)
                    out_fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.warning("AESE CLI: failed to re-write output with resolved names: %s", exc)

    # --- Human-readable sidecar (.txt) ---
    if args.format == "human" and all_events:
        txt_path = os.path.splitext(args.output)[0] + ".txt"
        try:
            human_text = render_event_log(all_events)
            with open(txt_path, "w", encoding="utf-8") as txt_fh:
                txt_fh.write(human_text + "\n")
            logger.info("AESE CLI: human-readable log written to %s", txt_path)
            print(f"  Human-readable log: {txt_path}")
        except Exception as exc:
            logger.warning("Failed to write human-readable log: %s", exc)

    logger.info(
        "AESE CLI done. %d events written to %s", event_count, args.output
    )


if __name__ == "__main__":
    main()
