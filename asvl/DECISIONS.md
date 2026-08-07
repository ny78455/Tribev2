# DECISIONS.md
# Engineering Assumptions & Decisions

This file documents every assumption and design decision made during ASVL Module 1 implementation,
as required by the engineering contract (§0 Role Instruction).

---

## 1. Platform: Windows — `psutil` instead of `resource.getrusage`

**Decision:** `resource.getrusage` is a POSIX-only call unavailable on Windows.
Replaced with `psutil.Process().memory_info().rss` throughout `pipeline.py` and `eval/run_eval.py`.
`psutil` is added to `requirements.txt` (pinned `>=5.9.0`).

**Impact:** Peak RSS reporting is fully functional on Windows, macOS, and Linux.
If `psutil` is not installed, the pipeline logs a warning and reports `-1.0` MB (graceful degradation).

---

## 2. Test Videos: Synthetic Generation via PyAV

**Decision:** No sample video was provided. All acceptance tests and integration tests
generate synthetic MP4 files using PyAV directly (not `cv2.VideoCapture`) into `tmp_path`.
Synthetic videos use H.264 (`libx264`) with solid-color or random-noise frames
to exercise the "static vs. motion" adaptive behavior.

**Impact:** All tests are self-contained and require no external assets.
A real film can be substituted in any test by replacing `_make_synthetic_video()` with
the actual path.

---

## 3. Speech/Music Classification: Heuristic, Not a Classifier

**Decision:** Per §5.3 contract, `speech_prob` and `music_prob` in `AudioFeatures` are
computed via a simple heuristic combining:
  - **Zero-Crossing Rate (ZCR)**: Speech typically exhibits moderate ZCR (0.03–0.20).
  - **Spectral Centroid**: Speech has a lower spectral centroid (< 3500 Hz) than music.

Speech probability is high when both conditions hold; music probability is high when
centroid is elevated (≥ 1000 Hz) and ZCR is outside the speech range.

**Rationale:** No trained classifier is used (contract §0 explicitly forbids deep learning).
The heuristic is documented as approximate — not suitable for precision speech/music segmentation.
A production system should replace this with a lightweight trained classifier (e.g. YAMNet)
in a future module.

---

## 4. Novelty V1: Histogram-Based (Not Embedding-Based)

**Decision:** `features/novelty.py` uses chi-squared distance between the current frame's
HSV histogram and the rolling buffer's mean HSV histogram. This is the V1 baseline.

A `# TODO: swap for embedding-based novelty` comment is placed in `novelty.py` to mark
the upgrade point for a future module using CLIP or CNN embeddings.

**Normalization:** Chi-squared distance is normalized by an empirical max of 3.0, which
corresponds to completely disjoint normalized histograms over 3 channels × 64 bins.

---

## 5. HDR / 4K Input Handling

**Decision:** PyAV's `to_ndarray(format="rgb24")` handles HDR tone-mapping implicitly
for most codecs (H.265, VP9, AV1) when decoded to 8-bit RGB. Explicit HDR-to-SDR
conversion (e.g. BT.2020→BT.709 gamut mapping) is NOT implemented in Module 1.

**Status:** Tested on synthetic 320×240 frames. 1080p/4K/HDR clips are **untested**
(documented in README.md). If HDR tone-mapping produces visually incorrect results,
a `libav` filter graph (`zscale` + `tonemap`) should be inserted in `decoder.py`.

---

## 6. Audio Extraction: PyAV First, ffmpeg subprocess Fallback

**Decision:** `audio.py` attempts PyAV audio demux first. If the audio stream is in a
format PyAV cannot decode to a numpy array directly (e.g. certain AAC variants), it
falls back to spawning an `ffmpeg` subprocess to write a temporary WAV, then loads via
`librosa.load()`.

**Impact:** Requires `ffmpeg` to be on `PATH` for the fallback path. If neither PyAV nor
ffmpeg can extract audio, the pipeline gracefully continues with `SILENT_AUDIO` defaults.

---

## 7. Audio Index Pre-computation

**Decision:** Audio features are pre-computed once at pipeline startup (`build_audio_index()`)
and indexed by timestamp for O(1) lookup during the frame decode loop. This trades a
small upfront cost (~5–30s for a 2-hour film at 250ms windows) for zero I/O in the hot path.

**Memory impact:** At 250ms windows, a 2-hour film produces ~28,800 `AudioFeatures` objects.
Each holds a `(13,)` float32 MFCC array + 4 floats + 1 bool ≈ ~150 bytes. Total: ~4.3 MB.
Well within the 2GB memory budget.

---

## 8. Subtitle Density Window

**Decision:** `subtitle_density()` uses a fixed 5000ms (5s) lookahead window as the default.
This was chosen to capture whether the *upcoming* few seconds are subtitle-heavy,
which informs the controller that dialogue is about to occur and frames should be retained.
The window size is a parameter and can be overridden by callers.

---

## 9. Optical Flow Normalization Constant

**Decision:** The Farneback optical flow magnitude is normalized by `_EMPIRICAL_MAX_MAGNITUDE = 50.0`
pixels/frame. This was chosen based on typical dense-flow magnitudes for camera pans and action
sequences in 24fps HD video. For slow-motion or 4K video, this constant may need tuning.

---

## 10. Scheduler Reset After Non-Monotonic Input

**Decision:** The scheduler silently drops frames whose `timestamp_ms <= last_emitted_ts`
rather than raising an exception. This is consistent with the decoder's behavior of
skipping non-monotonic PTS values — the pipeline should degrade gracefully, not crash,
on malformed input.

---

## 11. No GPU Dependency

**Decision:** All optical flow and feature computation uses OpenCV CPU backends.
`cv2.cuda` optical flow is mentioned in the contract as optional; it is NOT implemented
in Module 1 to keep the dependency surface minimal. A TODO comment in `motion.py`
marks the GPU upgrade path.

---

## 12. v1.1 Fix — Motion Score: p90 Replaces Mean, Normalization Constant 50.0 → 15.0

**Defect (v1.0):** `compute_motion_score` used `np.mean(magnitude) / 50.0`. Global mean
dilutes a fast-moving subject (e.g. sprinting actor, rolling car) against a large static
background. Max observed score on the high-action segment was 0.397 — never reaching the
0.6 importance threshold for the 5 fps tier.

**Fix:**
- **Aggregation:** `np.mean(magnitude)` → `np.percentile(magnitude, 90)` (p90 of pixel
  magnitudes). A moving subject occupying ~10% of the frame now drives the score rather
  than being averaged away.
- **Normalization constant:** `_EMPIRICAL_MAX_MAGNITUDE = 50.0` (mean-based) →
  `MOTION_NORM_CONST = 15.0` (p90-based). At 30fps on 720×1280 video, p90 in a
  fast-action frame typically sits in 10–20 px/frame; 15.0 maps that range to 0.67–1.0.
  Static/dialogue scenes have p90 ≈ 1–3 px/frame → score 0.07–0.20, safely below all
  action tiers.

**Old formula:** `score = min(mean_magnitude / 50.0, 1.0)`
**New formula:** `score = min(np.percentile(magnitude, 90) / 15.0, 1.0)`

**Validation:** Synthetic unit test (100×100 moving block against static 720×1280 bg):
new score > 0.5, old score < 0.15 on identical input.

---

## 13. v1.1 Fix — Scene Threshold: hist_dist Formula Corrected + Threshold 0.7 → 0.45

**Defect (v1.0):** `scene_change` was `False` on 100% of packets despite 4 known hard
cuts. Two compounding issues:

1. **hist_dist formula:** `(1.0 - correlation) / 2.0` silently halved the histogram
   distance signal. Histogram correlation is in [-1, 1], so a hard cut producing
   correlation = −0.5 yielded `hist_dist = 0.75`. But dividing by 2 gave only 0.375,
   well below both the old (0.7) and corrected (0.45) threshold.

2. **Threshold:** Even with the formula fixed, a threshold of 0.7 is aggressive for
   combined hist+SSIM distance. Real hard cuts on this clip produce combined scores
   in the 0.5–0.8 range; 0.45 reliably detects all 4 known cuts without excessive
   false positives in static segments.

**Fix:**
- **Old:** `hist_dist = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))`
- **New:** `hist_dist = float(np.clip(1.0 - max(0.0, correlation), 0.0, 1.0))`
  (Negative correlation → clips to 0 before inversion; result ∈ [0, 1], matching
  SSIM distance scale.)

- **Old threshold:** `scene_threshold = 0.7` (in config.default.yaml, types.py,
  scene.py, scene_transition.py)
- **New threshold:** `scene_threshold = 0.45`

**Files changed:** `asvl/features/scene.py`, `asvl/asvl/types.py`,
`asvl/scene_transition.py`, `config.default.yaml`

---

## 14. v1.1 Fix — Novelty Saturation: Empty-Buffer Default 1.0 → 0.0 + Distance Metric Swap

**Defect (v1.0):** `novelty_score` was exactly 1.0 on all 49/49 emitted packets. Two
root causes:

1. **Empty-buffer cold-start:** `compute_novelty(..., buffer_mean_hist=None)` returned
   `1.0` (maximally novel). During the first `buffer_seconds` of video, the buffer is
   empty on many frames, producing constant 1.0 before the buffer fills up.
   Additionally, `pipeline.py` had `else 1.0` as the fallback when `buf_hist is None`.

2. **Distance metric:** Chi-squared distance with empirical normalization cap of 3.0
   could saturate on frames that differ significantly in histogram shape, keeping the
   score pegged at 1.0 even after the buffer populated.

**Fix (novelty.py):**
- **Old:** `if buffer_mean_hist is None ...: return 1.0`
- **New:** `if buffer_mean_hist is None ...: return 0.0`
  "No reference → not novel" is the correct safe default; it avoids flooding the
  importance formula with spurious novelty during cold-start.

- **Old metric:** Chi-squared distance, normalized by empirical cap 3.0 (saturates)
- **New metric:** `1.0 - max(0.0, cv2.compareHist(..., HISTCMP_CORREL))` — correlation
  in [-1, 1], naturally bounded to [0, 1] after clipping and inversion; no empirical
  cap needed.

**Fix (pipeline.py):**
- Line 154: `else 1.0` → `else 0.0` for consistency with the novelty.py change.

---

## 15. v1.1 — Per-Stage Timing Instrumentation Added to pipeline.py

**Change:** Added `time.perf_counter()` timing wrappers around each of the six pipeline
stages (decode, motion, scene, audio, novelty, schedule). Totals are logged at end-of-run
at INFO level:

```
ASVL stage timings (total seconds) — decode=Xs  motion=Xs  scene=Xs  audio=Xs  novelty=Xs  schedule=Xs
```

**Purpose:** Enables regression assertion that `audio` stage total < 5s for a 60s clip
(audio features are pre-built once via `build_audio_index()` — O(1) lookup, no I/O in
the hot path).

## 16. v1.1 Performance — Thumbnail Downsampling for SSIM and Optical Flow

**Problem:** On a 720×1280 clip at 30fps, profiling revealed:
- SSIM (full-res 720×1280 grayscale): ~93ms/frame
- Farneback optical flow (full-res): ~134ms/frame
- Total: ~227ms/frame × 1800 frames ≈ 409 seconds → well over 60s budget.

**Fix: compute both on 90px short-edge thumbnails.**

| Operation | Full-res | Thumbnail (90px) | Speedup |
|-----------|----------|------------------|---------|
| SSIM      | 92.6 ms  | 2.1 ms           | 44×     |
| Farneback | 133.8 ms | 3.3 ms           | 41×     |

**Rationale:** Scene cuts are global appearance changes; histogram distances and
structural similarity at thumbnail resolution are sufficient to distinguish them.
Optical flow p90 is a ratio (subject pixels / total pixels) — the ratio is
preserved regardless of resolution. Magnitudes scale proportionally, requiring
only a constant adjustment:

- `MOTION_NORM_CONST`: `15.0` (full-res p90) → `2.0` (thumbnail p90, ~0.125× scale)

**Files changed:** `asvl/features/motion.py` (`_FLOW_SHORT_EDGE = 90`, `MOTION_NORM_CONST = 2.0`),
`asvl/features/scene.py` (`_SSIM_SHORT_EDGE = 90`, `_thumbnail_gray()` helper added).

**Estimated per-frame budget at thumbnail:** ~5–10ms (motion + scene + novelty + audio lookup)
→ 1800 frames × 10ms ≈ 18s. Well within the 60s limit.

---

## 17. v1.1 Performance — Incremental Histogram Sum in RollingFrameBuffer

**Problem:** `buffer.mean_histogram()` was called once per frame (1800× for a 60s clip)
and computed histograms for ALL frames currently in the buffer. With `buffer_seconds=10`
at 30fps, that's up to 300 frames × 1.9ms/hist = 570ms per call, growing as the buffer fills.

**Fix (buffer.py):**
- Pre-compute each frame's histogram in `push()` (once per frame, at thumbnail resolution).
- Maintain a running `_hist_sum` array: add the new histogram on push, subtract the
  evicted histogram when the buffer is at capacity.
- `mean_histogram()` returns `_hist_sum / len(buffer)` — O(1) regardless of buffer size.

**Fix (novelty.py):**
- `_frame_histogram()` updated to use the same 90px thumbnail so comparisons between
  the current frame's hist and the buffer mean hist are at the same resolution.

**Impact:** `mean_histogram()` goes from O(N) to O(1). Per-frame latency stabilizes.
