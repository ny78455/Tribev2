"""
eval/compare_fixed_vs_adaptive.py
Comparison harness: ASVL adaptive vs. fixed-FPS modes (§9).

Outputs a markdown table matching §21's "Efficiency" table structure.
QA Accuracy and downstream-metric columns are marked N/A — those require
a downstream VLM module not in scope for ASVL itself.

Usage:
    python eval/compare_fixed_vs_adaptive.py --input movie.mp4 [--config config.yaml]
"""
import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.run_eval import run_eval

logging.basicConfig(level=logging.WARNING)


def build_comparison_table(results: list) -> str:
    """
    Build a markdown table from a list of result dicts.

    Columns:
        Mode | Frames Kept | Avg FPS Used | Wall-clock (s) | Peak RSS (MB) | QA Accuracy
    """
    header = (
        "| Mode | Frames Kept | Avg FPS Used | Wall-clock (s) | Peak RSS (MB) | QA Accuracy |\n"
        "|------|-------------|-------------|----------------|---------------|-------------|\n"
    )
    rows = []
    for r in results:
        qa = "N/A (*)"
        row = (
            f"| {r['label']} | {r['frames_kept']} | {r['avg_fps_used']} | "
            f"{r['wall_clock_s']} | {r['peak_rss_mb']} | {qa} |"
        )
        rows.append(row)

    table = header + "\n".join(rows)
    table += (
        "\n\n(*) QA Accuracy requires a downstream VLM evaluation module — "
        "out of scope for ASVL Module 1."
    )
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Compare ASVL adaptive vs. fixed-FPS modes."
    )
    parser.add_argument("--input", required=True, metavar="VIDEO", help="Input video path.")
    parser.add_argument("--config", default=None, metavar="YAML", help="Config path.")
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write markdown table to this file (default: print to stdout).",
    )
    args = parser.parse_args()

    print(f"Evaluating: {args.input}")
    print("Running adaptive mode...")
    results = [run_eval(args.input, args.config, label="Adaptive (ASVL)")]

    for fixed_fps in [1, 2, 5]:
        print(f"Running fixed {fixed_fps}fps mode...")
        results.append(
            run_eval(
                args.input,
                args.config,
                label=f"Fixed {fixed_fps}fps",
                fixed_fps=float(fixed_fps),
            )
        )

    table = build_comparison_table(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# ASVL Efficiency Comparison\n\n")
            f.write(table)
        print(f"Table written to: {args.output}")
    else:
        print("\n## ASVL Efficiency Comparison\n")
        print(table)


if __name__ == "__main__":
    main()
