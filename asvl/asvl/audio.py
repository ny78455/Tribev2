"""
asvl/asvl/audio.py
Audio feature extraction aligned to video timestamps.

Extracts audio from a video file via PyAV (preferred) with ffmpeg subprocess fallback.
Uses 250ms analysis windows; yields AudioFeatures aligned to timestamp_ms.

Speech/music classification is a HEURISTIC based on zero-crossing rate and spectral
centroid banding — NOT a trained classifier. See DECISIONS.md for rationale.
"""
import io
import logging
import subprocess
import tempfile
from typing import Iterator, Optional

import librosa
import numpy as np

from .types import AudioFeatures

logger = logging.getLogger(__name__)

# Window size for analysis
_WINDOW_MS = 250
# Silence threshold — frames with RMS below this are considered silent
_SILENCE_RMS_THRESHOLD = 0.02
# Speech: ZCR typically 0.05–0.15 at 22050Hz, spectral centroid < 3000Hz
_SPEECH_ZCR_RANGE = (0.03, 0.20)
_SPEECH_CENTROID_MAX = 3500.0
# Music: higher centroid, moderate ZCR
_MUSIC_CENTROID_MIN = 1000.0


def _extract_audio_numpy(video_path: str) -> Optional[tuple]:
    """
    Extract audio from the video file.

    Tries PyAV first; falls back to ffmpeg subprocess.

    Returns:
        (audio_array: np.ndarray float32 mono, sample_rate: int)
        or None if no audio track found.
    """
    # --- Try PyAV ---
    try:
        import av as pyav

        container = pyav.open(video_path)
        if not container.streams.audio:
            container.close()
            raise ValueError("No audio streams")

        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.sample_rate
        chunks = []

        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            # Convert to mono float32
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))

        container.close()

        if not chunks:
            return None

        audio = np.concatenate(chunks)
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio, sample_rate

    except Exception as exc:
        logger.warning("PyAV audio extraction failed (%s); trying ffmpeg subprocess.", exc)

    # --- Fallback: ffmpeg subprocess to wav ---
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "22050", "-ac", "1",
                tmp_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("ffmpeg subprocess failed: %s", result.stderr.decode())
            return None

        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        return audio, sr

    except Exception as exc:
        logger.error("ffmpeg audio extraction failed: %s", exc)
        return None


def _compute_speech_music_probs(
    y_window: np.ndarray,
    sr: int,
) -> tuple:
    """
    Heuristic speech/music classification.

    Speech indicators:
      - Moderate ZCR (0.03–0.20)
      - Lower spectral centroid (< 3500 Hz)

    Music indicators:
      - Any spectral centroid ≥ 1000 Hz
      - ZCR outside the speech range

    Returns:
        (speech_prob: float, music_prob: float)
    """
    if len(y_window) < 64:
        return 0.0, 0.0

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_window)))
    spec_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=y_window, sr=sr))
    )

    # Speech probability: high if ZCR in range AND centroid below threshold
    speech_zcr_ok = _SPEECH_ZCR_RANGE[0] <= zcr <= _SPEECH_ZCR_RANGE[1]
    speech_centroid_ok = spec_centroid < _SPEECH_CENTROID_MAX

    if speech_zcr_ok and speech_centroid_ok:
        speech_prob = 0.75
    elif speech_zcr_ok or speech_centroid_ok:
        speech_prob = 0.4
    else:
        speech_prob = 0.1

    # Music probability: inversely weighted
    music_centroid_ok = spec_centroid >= _MUSIC_CENTROID_MIN
    zcr_non_speech = not speech_zcr_ok

    if music_centroid_ok and zcr_non_speech:
        music_prob = 0.8
    elif music_centroid_ok:
        music_prob = 0.5
    else:
        music_prob = 0.2

    return float(speech_prob), float(music_prob)


def extract_audio_features(
    video_path: str,
    window_ms: int = _WINDOW_MS,
) -> Iterator[AudioFeatures]:
    """
    Generator yielding AudioFeatures aligned to timestamps (in ms).

    Each yielded AudioFeatures corresponds to a 250ms window of audio,
    with the timestamp at the START of that window.

    Args:
        video_path: Path to the video file.
        window_ms: Analysis window size in milliseconds (default 250).

    Yields:
        AudioFeatures (energy, mfcc, spectral_flux, speech_prob, music_prob, silence)

    Note:
        Yields nothing if no audio track is found (graceful degradation).
    """
    result = _extract_audio_numpy(video_path)
    if result is None:
        logger.warning("No audio found in %s; audio features will be empty.", video_path)
        return

    audio, sr = result
    window_samples = int(sr * window_ms / 1000)

    if window_samples <= 0:
        return

    # Onset envelope for spectral flux (full signal, hop=window_samples)
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=window_samples
    )

    # Global RMS max for normalization
    global_rms_max = float(np.sqrt(np.mean(audio ** 2))) * 5.0 + 1e-9

    for i, start in enumerate(range(0, len(audio) - window_samples + 1, window_samples)):
        window = audio[start: start + window_samples]

        # RMS energy (normalized 0-1)
        rms_val = float(np.sqrt(np.mean(window ** 2)))
        energy = float(np.clip(rms_val / global_rms_max, 0.0, 1.0))

        # MFCC (13 coefficients)
        try:
            mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)  # shape (13,)
        except Exception:
            mfcc_mean = np.zeros(13, dtype=np.float32)

        # Spectral flux from onset envelope
        flux_val = float(onset_env[i]) if i < len(onset_env) else 0.0
        spectral_flux = float(np.clip(flux_val / (np.max(onset_env) + 1e-9), 0.0, 1.0))

        # Silence
        silence = rms_val < _SILENCE_RMS_THRESHOLD

        # Speech / music heuristic
        speech_prob, music_prob = _compute_speech_music_probs(window, sr)

        yield AudioFeatures(
            energy=energy,
            mfcc=mfcc_mean.astype(np.float32),
            spectral_flux=spectral_flux,
            speech_prob=speech_prob,
            music_prob=music_prob,
            silence=silence,
        )


def build_audio_index(video_path: str, window_ms: int = _WINDOW_MS) -> list:
    """
    Pre-compute all AudioFeatures and store as a list for O(1) lookup by timestamp.

    Returns:
        List of (start_ms: float, end_ms: float, AudioFeatures) tuples.
    """
    entries = []
    for i, af in enumerate(extract_audio_features(video_path, window_ms)):
        start_ms = i * window_ms
        end_ms = start_ms + window_ms
        entries.append((float(start_ms), float(end_ms), af))
    return entries


def get_audio_at(audio_index: list, timestamp_ms: float) -> Optional[AudioFeatures]:
    """
    Binary-search the audio index for the window containing timestamp_ms.

    Args:
        audio_index: Output of build_audio_index().
        timestamp_ms: Query timestamp.

    Returns:
        AudioFeatures for the window containing timestamp_ms, or None.
    """
    lo, hi = 0, len(audio_index) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, af = audio_index[mid]
        if start <= timestamp_ms < end:
            return af
        elif timestamp_ms < start:
            hi = mid - 1
        else:
            lo = mid + 1
    # Return last window if beyond end
    if audio_index and timestamp_ms >= audio_index[-1][0]:
        return audio_index[-1][2]
    return None


# Default silent AudioFeatures for when no audio is available
SILENT_AUDIO = AudioFeatures(
    energy=0.0,
    mfcc=np.zeros(13, dtype=np.float32),
    spectral_flux=0.0,
    speech_prob=0.0,
    music_prob=0.0,
    silence=True,
)
