"""
asvl/asvl/transcribe.py
Auto-transcription: whisper.cpp native binary OR faster-whisper Python package.

When no subtitle file is supplied, this module tries two backends in order:
  1. whisper.cpp native binary (whisper-cli / whisper / main)
     Fast, no GPU needed. Requires whisper.cpp installed separately.
  2. faster-whisper Python package  (pip install faster-whisper)
     Pure Python, downloads model on first use. Slower on CPU but needs
     no native build.

If both are unavailable the pipeline runs without subtitles (graceful).

Environment variables
---------------------
WHISPER_BIN       : Override whisper binary path (full path to executable).
WHISPER_MODEL     : Override model file path (full path to .bin).
WHISPER_MODEL_DIR : Directory to search for ggml model files.
FASTER_WHISPER_MODEL : faster-whisper model size or HF repo id
                       (default: "base"; options: tiny, base, small, medium, large-v3).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binary search
# ---------------------------------------------------------------------------

# Candidate executable names, tried in order
_WHISPER_BINARY_NAMES = ["whisper-cli", "whisper", "main"]

# Common installation directories to check beyond PATH
_WHISPER_SEARCH_DIRS = [
    # Windows — common cmake build outputs
    r"C:\whisper.cpp\build\bin\Release",
    r"C:\whisper.cpp\build\bin",
    r"C:\whisper.cpp\build",
    r"C:\Program Files\whisper.cpp\bin",
    # Linux / macOS
    "/usr/local/bin",
    "/opt/whisper.cpp/bin",
    str(Path.home() / "whisper.cpp" / "build" / "bin"),
]

# Candidate model filenames, tried in order (smallest → largest)
_MODEL_FILENAMES = [
    "ggml-base.en.bin",
    "ggml-base.bin",
    "ggml-small.en.bin",
    "ggml-small.bin",
    "ggml-medium.en.bin",
    "ggml-medium.bin",
    "ggml-large-v3.bin",
]

# Directories to search for model files
def _model_search_dirs(binary_path: Optional[str] = None) -> list[str]:
    dirs: list[str] = []
    if "WHISPER_MODEL_DIR" in os.environ:
        dirs.append(os.environ["WHISPER_MODEL_DIR"])
    if binary_path:
        # <binary_dir>/models/ and <binary_dir>/../models/
        bp = Path(binary_path).resolve()
        dirs.append(str(bp.parent / "models"))
        dirs.append(str(bp.parent.parent / "models"))
    try:
        home = Path.home()
        dirs.append(str(home / ".cache" / "whisper"))
        dirs.append(str(home / "whisper.cpp" / "models"))
    except RuntimeError:
        pass  # home dir unavailable (e.g. stripped env in tests)
    dirs.append(r"C:\whisper.cpp\models")
    dirs.append("/usr/local/share/whisper/models")
    return dirs


def find_whisper_binary() -> Optional[str]:
    """
    Locate the whisper.cpp CLI executable.

    Search order:
    1. ``WHISPER_BIN`` environment variable (full path).
    2. Each name in ``_WHISPER_BINARY_NAMES`` via ``shutil.which`` (PATH).
    3. Each name × each directory in ``_WHISPER_SEARCH_DIRS``.

    Returns:
        Absolute path to the binary, or None if not found.
    """
    # 1. Env-var override
    env_bin = os.environ.get("WHISPER_BIN")
    if env_bin:
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            logger.debug("whisper binary from WHISPER_BIN: %s", env_bin)
            return env_bin
        logger.warning(
            "WHISPER_BIN is set to '%s' but file is not executable — ignoring.", env_bin
        )

    # 2. PATH lookup
    for name in _WHISPER_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            logger.debug("whisper binary found in PATH: %s", found)
            return found

    # 3. Well-known directories
    exts = [".exe", ""] if os.name == "nt" else [""]
    for directory in _WHISPER_SEARCH_DIRS:
        for name in _WHISPER_BINARY_NAMES:
            for ext in exts:
                candidate = os.path.join(directory, name + ext)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    logger.debug("whisper binary found at: %s", candidate)
                    return candidate

    logger.debug("whisper.cpp binary not found — auto-transcription unavailable.")
    return None


def find_whisper_model(binary_path: Optional[str] = None) -> Optional[str]:
    """
    Locate a ggml model file for whisper.cpp.

    Search order:
    1. ``WHISPER_MODEL`` environment variable (full path to .bin).
    2. Each model filename × each directory returned by ``_model_search_dirs``.

    Returns:
        Absolute path to the model file, or None if not found.
    """
    # 1. Env-var override
    env_model = os.environ.get("WHISPER_MODEL")
    if env_model:
        if os.path.isfile(env_model):
            logger.debug("whisper model from WHISPER_MODEL: %s", env_model)
            return env_model
        logger.warning(
            "WHISPER_MODEL is set to '%s' but file not found — ignoring.", env_model
        )

    # 2. Search directories
    for directory in _model_search_dirs(binary_path):
        for model_name in _MODEL_FILENAMES:
            candidate = os.path.join(directory, model_name)
            if os.path.isfile(candidate):
                logger.debug("whisper model found: %s", candidate)
                return candidate

    logger.debug(
        "No whisper model found. Set WHISPER_MODEL or WHISPER_MODEL_DIR. "
        "Download models from: https://huggingface.co/ggerganov/whisper.cpp"
    )
    return None


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, wav_path: str) -> bool:
    """
    Extract a 16kHz mono WAV from *video_path* into *wav_path*.

    Tries ffmpeg first (fast); falls back to PyAV if ffmpeg is not in PATH.

    Returns:
        True on success, False on failure.
    """
    # Try ffmpeg
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-vn",
                    wav_path,
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.debug("Audio extracted via ffmpeg to %s", wav_path)
                return True
            logger.warning(
                "ffmpeg exited %d: %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:200],
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out during audio extraction.")
        except Exception as exc:
            logger.warning("ffmpeg failed: %s — trying PyAV fallback.", exc)

    # PyAV fallback
    try:
        import av
        import numpy as np
        import wave, array

        container = av.open(video_path)
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            logger.warning("No audio stream found in %s.", video_path)
            return False

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm_data: list[bytes] = []

        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for rf in (resampled if isinstance(resampled, list) else [resampled]):
                    pcm_data.append(bytes(rf.planes[0]))

        container.close()

        if not pcm_data:
            logger.warning("No audio data decoded from %s.", video_path)
            return False

        raw = b"".join(pcm_data)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(16000)
            wf.writeframes(raw)

        logger.debug("Audio extracted via PyAV to %s", wav_path)
        return True

    except Exception as exc:
        logger.warning("PyAV audio extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# faster-whisper Python fallback
# ---------------------------------------------------------------------------

def _transcribe_faster_whisper(
    video_path: str,
    model_name: str = "base",
    language: str = "en",
) -> Optional[str]:
    """
    Transcribe *video_path* using the faster-whisper Python package.

    Downloads the model from HuggingFace on first use (cached in
    ~/.cache/huggingface/).

    Args:
        video_path:  Path to the input video.
        model_name:  Model size or HF repo id (default: "base").
                     Options: tiny, base, small, medium, large-v3
        language:    Language code ("en", "auto", etc.)

    Returns:
        Path to generated .srt file, or None on failure.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        logger.warning(
            "faster-whisper not installed — no transcription backend available. "
            "Run: pip install faster-whisper"
        )
        return None

    tmp_dir = tempfile.mkdtemp(prefix="asvl_whisper_")
    srt_path = os.path.join(tmp_dir, "transcript.srt")

    try:
        logger.info(
            "Auto-transcription (faster-whisper): loading model '%s' …", model_name
        )
        # device="auto" uses CUDA if available, else CPU
        fw_language = None if language == "auto" else language
        model = WhisperModel(model_name, device="auto", compute_type="auto")

        logger.info(
            "Auto-transcription (faster-whisper): transcribing %s …",
            os.path.basename(video_path),
        )
        segments, info = model.transcribe(
            video_path,
            language=fw_language,
            beam_size=5,
            vad_filter=True,
        )
        logger.info(
            "Detected language '%s' (prob=%.2f)",
            info.language,
            info.language_probability,
        )

        # Write SRT
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = _seconds_to_srt_time(seg.start)
                end = _seconds_to_srt_time(seg.end)
                f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")

        if os.path.getsize(srt_path) == 0:
            logger.warning("faster-whisper produced an empty SRT — skipping.")
            return None

        logger.info(
            "Auto-transcription complete: %s (%.1f KB)",
            srt_path,
            os.path.getsize(srt_path) / 1024,
        )
        return srt_path

    except Exception as exc:
        logger.warning("faster-whisper transcription error: %s", exc, exc_info=True)
        return None


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp: HH:MM:SS,mmm"""
    ms = int(seconds * 1000)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main transcription entry point
# ---------------------------------------------------------------------------

def transcribe(
    video_path: str,
    model_path: Optional[str] = None,
    language: str = "en",
    binary_path: Optional[str] = None,
) -> Optional[str]:
    """
    Transcribe *video_path* to an SRT file using whisper.cpp.

    Args:
        video_path:   Path to the input video.
        model_path:   Override path to the ggml model (.bin). If None,
                      ``find_whisper_model()`` is called automatically.
        language:     Language code for whisper (e.g. "en", "auto").
        binary_path:  Override path to whisper-cli binary. If None,
                      ``find_whisper_binary()`` is called automatically.

    Returns:
        Path to the generated .srt file (lives in a temp directory that the
        caller owns), or None if transcription could not complete.

    Note:
        The returned .srt path is inside a ``tempfile.mkdtemp()`` directory.
        The caller is responsible for cleanup (or simply let the OS clean up
        on process exit — temp dirs are small).
    """
    # --- Locate binary: try whisper.cpp, then faster-whisper Python fallback ---
    binary = binary_path or find_whisper_binary()
    if binary is None:
        logger.info(
            "whisper.cpp not found — trying faster-whisper Python fallback."
        )
        return _transcribe_faster_whisper(
            video_path,
            model_name=os.environ.get("FASTER_WHISPER_MODEL", "base"),
            language=language,
        )

    # --- Locate model (whisper.cpp path) ---
    model = model_path or find_whisper_model(binary)
    if model is None:
        logger.warning(
            "No whisper.cpp model found — trying faster-whisper Python fallback. "
            "To use whisper.cpp: download a model and set WHISPER_MODEL or WHISPER_MODEL_DIR."
        )
        return _transcribe_faster_whisper(
            video_path,
            model_name=os.environ.get("FASTER_WHISPER_MODEL", "base"),
            language=language,
        )

    # --- Work in a temp directory ---
    tmp_dir = tempfile.mkdtemp(prefix="asvl_whisper_")
    wav_path = os.path.join(tmp_dir, "audio.wav")
    # whisper-cli --output-file <prefix> produces <prefix>.srt
    srt_prefix = os.path.join(tmp_dir, "transcript")
    srt_path = srt_prefix + ".srt"

    try:
        # Extract audio
        logger.info("Auto-transcription: extracting audio from %s …", os.path.basename(video_path))
        if not extract_audio(video_path, wav_path):
            logger.warning("Auto-transcription: audio extraction failed — skipping.")
            return None

        # Run whisper-cli
        cmd = [
            binary,
            "--model", model,
            "--language", language,
            "--output-srt",
            "--output-file", srt_prefix,
            "--file", wav_path,
        ]
        logger.info("Auto-transcription: running whisper-cli …")
        logger.debug("Command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,  # 10-minute hard cap
        )

        if result.returncode != 0:
            stderr_snippet = result.stderr.decode(errors="replace")[:400]
            logger.warning(
                "whisper-cli exited %d — auto-transcription failed.\n%s",
                result.returncode,
                stderr_snippet,
            )
            return None

        if not os.path.isfile(srt_path):
            logger.warning(
                "whisper-cli completed but %s was not created — skipping.", srt_path
            )
            return None

        # Quick sanity: non-empty SRT
        if os.path.getsize(srt_path) == 0:
            logger.warning("whisper-cli produced an empty SRT — skipping.")
            return None

        logger.info(
            "Auto-transcription complete: %s (%.1f KB)",
            srt_path,
            os.path.getsize(srt_path) / 1024,
        )
        return srt_path

    except subprocess.TimeoutExpired:
        logger.warning("whisper-cli timed out (>600s) — auto-transcription skipped.")
        return None
    except Exception as exc:
        logger.warning("Auto-transcription error: %s", exc, exc_info=True)
        return None
