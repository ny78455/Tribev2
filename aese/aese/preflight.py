"""
aese/preflight.py
Pre-flight signal-richness audit for AESE.

Run audit_manifest() + print_report() immediately after loading the manifest,
before the expensive boundary-detection pass. This turns "single 81s event,
no idea why" into an immediate, readable explanation every time.

Public API:
    audit_manifest(frame_packets, detector_mode) -> SignalRichnessReport
    print_report(report) -> None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import FramePacket


@dataclass
class SignalRichnessReport:
    """
    Summary statistics about the signals present in a manifest before
    the segmentation pass runs.
    """
    total_seconds: float
    pct_scene_change: float        # percentage of packets with scene_change=True
    pct_dialogue_present: float    # percentage of packets with non-empty subtitle_text
    mean_motion: float
    max_motion: float
    pct_frames_with_image: float   # percentage of packets where image is not None
    detector_mode: str             # active character detector chain name


def audit_manifest(
    frame_packets: List[FramePacket],
    detector_mode: str,
) -> SignalRichnessReport:
    """
    Compute a signal-richness summary from a list of FramePackets.

    Args:
        frame_packets: All packets from the manifest (already loaded into memory).
        detector_mode: Active detector chain name (from get_effective_detector_chain()).

    Returns:
        A SignalRichnessReport dataclass instance.
    """
    n = len(frame_packets)
    if n == 0:
        return SignalRichnessReport(
            total_seconds=0.0,
            pct_scene_change=0.0,
            pct_dialogue_present=0.0,
            mean_motion=0.0,
            max_motion=0.0,
            pct_frames_with_image=0.0,
            detector_mode=detector_mode,
        )

    total_seconds = (
        (frame_packets[-1].timestamp_ms - frame_packets[0].timestamp_ms) / 1000.0
    )
    pct_scene_change = 100.0 * sum(1 for p in frame_packets if p.scene_change) / n
    pct_dialogue_present = 100.0 * sum(1 for p in frame_packets if p.subtitle_text) / n
    motions = [p.motion_score for p in frame_packets]
    mean_motion = sum(motions) / n
    max_motion = max(motions)
    pct_frames_with_image = 100.0 * sum(1 for p in frame_packets if p.image is not None) / n

    return SignalRichnessReport(
        total_seconds=total_seconds,
        pct_scene_change=pct_scene_change,
        pct_dialogue_present=pct_dialogue_present,
        mean_motion=mean_motion,
        max_motion=max_motion,
        pct_frames_with_image=pct_frames_with_image,
        detector_mode=detector_mode,
    )


def print_report(r: SignalRichnessReport) -> None:
    """
    Print a human-readable signal-richness pre-flight report to stdout.

    Includes ⚠ warnings when key signals (scene_change, dialogue) are entirely
    absent — turning "single low-importance mega-event" into an immediate,
    self-explaining diagnostic.

    Always prints unconditionally — not behind --verbose — because the whole
    purpose is to make silent failures visible.
    """
    print("=" * 70)
    print("  SIGNAL RICHNESS PRE-FLIGHT REPORT")
    print(f"  Duration:              {r.total_seconds:.1f}s")
    print(f"  scene_change=True:     {r.pct_scene_change:.1f}% of frames")
    print(f"  dialogue present:      {r.pct_dialogue_present:.1f}% of frames")
    print(f"  motion (mean/max):     {r.mean_motion:.3f} / {r.max_motion:.3f}")
    print(f"  frames with image:     {r.pct_frames_with_image:.1f}%")
    print(f"  character detector:    {r.detector_mode}")

    if r.pct_scene_change == 0.0:
        print("  \u26a0  No scene cuts detected anywhere in this clip. Either the video is")
        print("    genuinely a single continuous shot, or Module 1's scene_change")
        print("    detector needs recalibration (see prior ASVL fix contract).")

    if r.pct_dialogue_present == 0.0:
        print("  \u26a0  No dialogue/subtitle signal detected. Pass --subtitles if a")
        print("    transcript exists, or dialogue-based boundaries are structurally disabled.")

    if r.pct_frames_with_image == 0.0:
        print("  \u26a0  No pixel data available (manifest-replay mode without --video).")
        print("    Character count, scene label, and VLM-based signals are all disabled.")

    print("=" * 70)
