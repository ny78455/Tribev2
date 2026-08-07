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
