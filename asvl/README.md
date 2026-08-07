# ASVL — Adaptive Streaming Video Loader

**Module 1: Cognitive Movie Understanding System**

ASVL streams a video file and emits only the frames that matter, at a **variable rate** driven by motion, scene change, audio energy, subtitle density, and novelty — instead of a fixed FPS. It uses classical CV/DSP only (OpenCV, optical flow, histograms, SSIM, librosa).

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Requires Python 3.10+. ffmpeg must be on `PATH` (for audio extraction fallback).

---

## Quick Start

```bash
# Adaptive sampling — emit only important frames, write manifest
python cli.py --input movie.mp4 --output out/ --save-manifest

# Full options
python cli.py --input movie.mp4 --output out/ \
  --config config.default.yaml \
  --subtitles movie.srt \
  --mode sync \
  --save-frames \
  --save-manifest

# Fixed FPS comparison (override adaptive)
python cli.py --input movie.mp4 --output out_fixed/ \
  --min-fps 2 --max-fps 2
```

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--input` | Path to input video (`.mp4/.mkv/.avi/.mov/.webm`) |
| `--output` | Output directory |
| `--config` | Path to YAML config (default: built-in defaults) |
| `--subtitles` | Optional `.srt` or `.vtt` subtitle file |
| `--mode` | `sync` (default) or `async` scheduler mode |
| `--save-frames` | Dump kept frames as JPEGs in `<output>/frames/` |
| `--save-manifest` | Write `<output>/manifest.jsonl` (no raw images) |
| `--verbose / -v` | Enable DEBUG logging |
| `--min-fps` | Override `minimum_fps` |
| `--max-fps` | Override `maximum_fps` |

---

## Manifest Format

Each line in `manifest.jsonl` is a JSON object (no raw image data):

```json
{"frame_id": 523, "timestamp_ms": 1062150.0, "fps_used": 5.0, "motion_score": 0.94, "scene_change": false, "audio_energy": 0.81, "novelty_score": 0.72, "decision_reason": "Fast motion", "subtitle_text": null}
```

---

## Configuration

Edit `config.default.yaml` or pass `--config` to override:

```yaml
adaptive: true
minimum_fps: 0.5
maximum_fps: 10
motion_threshold: 0.4
scene_threshold: 0.7
audio_threshold: 0.6
novelty_threshold: 0.5
buffer_seconds: 10
weights:
  motion: 0.30
  scene: 0.25
  audio: 0.20
  subtitle: 0.10
  novelty: 0.15
```

---

## Python Library Usage

```python
from asvl.config import load_config
from asvl.pipeline import run

config = load_config("config.default.yaml")

for packet in run("movie.mp4", config, subtitle_path="movie.srt"):
    print(f"[{packet.timestamp_ms:.0f}ms] fps={packet.fps_used} reason={packet.decision_reason}")
    # packet.image is an HxWx3 uint8 RGB numpy array
```

---

## Running Tests

```bash
cd asvl/
pytest tests/ -v
```

All tests are self-contained — they generate synthetic videos using PyAV.
No external video file is required.

---

## Evaluation Harness

```bash
# Single-run metrics
python eval/run_eval.py --input movie.mp4

# Adaptive vs. fixed 1/2/5fps comparison table
python eval/compare_fixed_vs_adaptive.py --input movie.mp4 --output eval_results.md
```

The output is a markdown table. `QA Accuracy` is marked `N/A` — it requires a downstream
VLM evaluation module (out of scope for ASVL Module 1).

---

## Architecture

```
Decoder (PyAV)
    │ (frame, timestamp_ms)
    ▼
Rolling Frame Buffer
    │ mean_histogram()
    ▼
Feature Extractors
  ├── motion.py       (Farneback optical flow)
  ├── scene.py        (Histogram + SSIM)
  ├── edges.py        (Canny XOR)
  ├── blur.py         (Laplacian variance)
  └── novelty.py      (Histogram chi-sq vs. buffer)
    │
Audio (librosa, 250ms windows)
    │
Subtitles (pysrt / webvtt)
    │
Controller
  └── importance score → FPS tier → decision_reason
    │
Scene Transition Detector
    │
Scheduler (sync / async queue)
    │
FramePacket → caller / CLI
```

---

## Non-Functional Guarantees

| Requirement | How it's met |
|---|---|
| Latency <100ms/decision | Per-frame timing in pipeline.py; WARNING if exceeded |
| Memory <2GB | Rolling buffer only; peak RSS logged via psutil |
| Monotonic output timestamps | Hard invariant in scheduler.py |
| Never loads full video | Generator-based PyAV decode in decoder.py |

---

## Untested Configurations

- **1080p / 4K / HDR input**: Tested on synthetic 320×240 frames. HDR tone-mapping
  (BT.2020→BT.709) is not explicitly implemented — PyAV's 8-bit RGB output may apply
  implicit tone-mapping. See `DECISIONS.md §5`.
- **4+ hour movies**: Memory should remain flat (rolling buffer only). Not validated
  on a real 4hr file; use `--dry-run` (no `--save-frames`) and monitor RSS.

---

## Explicit Non-Goals (Module 1)

- No scene/object/character understanding (no VLM calls)
- No training pipeline or learned policy
- No video re-encoding / transcoding
- No GPU requirement (CPU fallback for all operations)
- No UI (CLI + Python library only)
