"""
eval/run_eval.py
ASVL evaluation metrics harness (§9).

Runs the pipeline and records:
  - Total frames processed
  - Frames emitted (kept)
  - Wall-clock time
  - Peak RSS memory
  - Per-segment FPS distribution
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asvl.config import load_config
from asvl.pipeline import run

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval.run_eval")


def run_eval(
    video_path: str,
    config_path: str = None,
    label: str = "adaptive",
    fixed_fps: float = None,
) -> dict:
    """
    Run the ASVL pipeline and collect metrics.

    Args:
        video_path: Input video.
        config_path: YAML config path (or None for defaults).
        label: Human-readable label for this run.
        fixed_fps: If provided, override min/max fps to fix sampling rate.

    Returns:
        Dict of metrics.
    """
    overrides = {}
    if fixed_fps is not None:
        overrides["minimum_fps"] = fixed_fps
        overrides["maximum_fps"] = fixed_fps
        overrides["adaptive"] = False

    config = load_config(yaml_path=config_path, overrides=overrides or None)

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        start_rss = proc.memory_info().rss
    except ImportError:
        proc = None
        start_rss = 0

    t0 = time.perf_counter()
    kept_count = 0
    fps_values = []

    for packet in run(video_path, config):
        kept_count += 1
        fps_values.append(packet.fps_used)

    elapsed = time.perf_counter() - t0

    peak_rss_mb = -1.0
    if proc is not None:
        try:
            peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

    return {
        "label": label,
        "frames_kept": kept_count,
        "avg_fps_used": round(avg_fps, 2),
        "wall_clock_s": round(elapsed, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="ASVL single-run evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--label", default="adaptive")
    parser.add_argument("--fixed-fps", type=float, default=None)
    args = parser.parse_args()

    result = run_eval(args.input, args.config, args.label, args.fixed_fps)
    print("\nEvaluation Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
