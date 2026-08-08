"""
asvl/tests/test_transcribe.py
Unit tests for the asvl.transcribe auto-transcription module.

These tests use mocking and never require whisper.cpp to be installed.
They validate graceful degradation only (find_* returns None → no crash).
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure asvl package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from asvl.transcribe import find_whisper_binary, find_whisper_model, transcribe


class TestFindWhisperBinary(unittest.TestCase):
    """find_whisper_binary() must return None gracefully when nothing is installed."""

    def test_returns_none_when_not_in_path_and_no_env(self):
        """No binary in PATH, no env var → returns None (no crash, no exception)."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.isfile", return_value=False),
        ):
            result = find_whisper_binary()
        self.assertIsNone(result)

    def test_respects_whisper_bin_env_var(self):
        """WHISPER_BIN set to an executable file → returns that path."""
        fake_bin = "/fake/path/whisper-cli"
        with (
            patch.dict(os.environ, {"WHISPER_BIN": fake_bin}),
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            result = find_whisper_binary()
        self.assertEqual(result, fake_bin)

    def test_whisper_bin_env_not_executable_falls_through(self):
        """WHISPER_BIN set but not executable → continues searching (returns None if nothing found)."""
        fake_bin = "/fake/path/not-executable"
        with (
            patch.dict(os.environ, {"WHISPER_BIN": fake_bin}),
            patch("os.path.isfile", side_effect=lambda p: p == fake_bin),
            patch("os.access", return_value=False),   # not executable
            patch("shutil.which", return_value=None),
        ):
            result = find_whisper_binary()
        self.assertIsNone(result)

    def test_finds_binary_via_shutil_which(self):
        """Binary discoverable via PATH (shutil.which) → returns it."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("shutil.which", side_effect=lambda name: "/usr/local/bin/whisper-cli" if name == "whisper-cli" else None),
        ):
            result = find_whisper_binary()
        self.assertEqual(result, "/usr/local/bin/whisper-cli")


class TestFindWhisperModel(unittest.TestCase):
    """find_whisper_model() must return None gracefully when no model is present."""

    def test_returns_none_when_no_model_found(self):
        """No model in any search dir, no env var → returns None."""
        env_overrides = {k: v for k, v in os.environ.items() if k not in ("WHISPER_MODEL", "WHISPER_MODEL_DIR")}
        with (
            patch.dict(os.environ, env_overrides, clear=True),
            patch("os.path.isfile", return_value=False),
        ):
            result = find_whisper_model()
        self.assertIsNone(result)

    def test_respects_whisper_model_env_var(self):
        """WHISPER_MODEL points to existing file → returns that path."""
        fake_model = "/models/ggml-base.en.bin"
        with (
            patch.dict(os.environ, {"WHISPER_MODEL": fake_model}),
            patch("os.path.isfile", return_value=True),
        ):
            result = find_whisper_model()
        self.assertEqual(result, fake_model)

    def test_whisper_model_env_missing_file_falls_through(self):
        """WHISPER_MODEL set but file missing → falls through to search (returns None)."""
        fake_model = "/models/missing.bin"
        with (
            patch.dict(os.environ, {"WHISPER_MODEL": fake_model}),
            patch("os.path.isfile", return_value=False),
        ):
            result = find_whisper_model()
        self.assertIsNone(result)

    def test_finds_model_in_search_dir(self):
        """Model file exists in one of the standard search dirs → found."""
        expected = str(Path.home() / ".cache" / "whisper" / "ggml-base.en.bin")
        def fake_isfile(p):
            return p == expected
        # Only remove whisper-specific vars, keep USERPROFILE/HOME for Path.home()
        env_overrides = {k: v for k, v in os.environ.items() if k not in ("WHISPER_MODEL", "WHISPER_MODEL_DIR")}
        with (
            patch.dict(os.environ, env_overrides, clear=True),
            patch("os.path.isfile", side_effect=fake_isfile),
        ):
            result = find_whisper_model()
        self.assertEqual(result, expected)


class TestTranscribeGracefulDegradation(unittest.TestCase):
    """transcribe() must return None without raising when components are unavailable."""

    def test_returns_none_when_binary_not_found(self):
        """No binary → transcribe() returns None immediately."""
        with patch("asvl.transcribe.find_whisper_binary", return_value=None):
            result = transcribe("/fake/video.mp4")
        self.assertIsNone(result)

    def test_returns_none_when_model_not_found(self):
        """Binary found but no model → transcribe() returns None."""
        with (
            patch("asvl.transcribe.find_whisper_binary", return_value="/usr/bin/whisper-cli"),
            patch("asvl.transcribe.find_whisper_model", return_value=None),
        ):
            result = transcribe("/fake/video.mp4")
        self.assertIsNone(result)

    def test_returns_none_when_audio_extraction_fails(self):
        """Binary + model found but audio extraction fails → returns None."""
        with (
            patch("asvl.transcribe.find_whisper_binary", return_value="/usr/bin/whisper-cli"),
            patch("asvl.transcribe.find_whisper_model", return_value="/models/ggml-base.en.bin"),
            patch("asvl.transcribe.extract_audio", return_value=False),
        ):
            result = transcribe("/fake/video.mp4")
        self.assertIsNone(result)

    def test_returns_none_when_whisper_subprocess_fails(self):
        """whisper-cli exits non-zero → returns None."""
        import subprocess
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"error: model file not found"
        with (
            patch("asvl.transcribe.find_whisper_binary", return_value="/usr/bin/whisper-cli"),
            patch("asvl.transcribe.find_whisper_model", return_value="/models/ggml-base.en.bin"),
            patch("asvl.transcribe.extract_audio", return_value=True),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = transcribe("/fake/video.mp4")
        self.assertIsNone(result)

    def test_returns_srt_path_on_success(self):
        """All components succeed → returns a path ending in .srt."""
        import subprocess, tempfile, os

        mock_result = MagicMock()
        mock_result.returncode = 0

        # We need the srt file to actually exist in the temp dir
        def fake_subprocess_run(cmd, **kwargs):
            # The srt prefix is the --output-file arg
            out_file_idx = cmd.index("--output-file") + 1
            srt_prefix = cmd[out_file_idx]
            srt_path = srt_prefix + ".srt"
            with open(srt_path, "w") as f:
                f.write("1\n00:00:01,000 --> 00:00:02,000\nHello world\n")
            return mock_result

        with (
            patch("asvl.transcribe.find_whisper_binary", return_value="/usr/bin/whisper-cli"),
            patch("asvl.transcribe.find_whisper_model", return_value="/models/ggml-base.en.bin"),
            patch("asvl.transcribe.extract_audio", return_value=True),
            patch("subprocess.run", side_effect=fake_subprocess_run),
        ):
            result = transcribe("/fake/video.mp4")

        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".srt"))
        # Cleanup
        if result and os.path.isfile(result):
            os.remove(result)


class TestPipelineGracefulWithNoWhisper(unittest.TestCase):
    """
    Verify the pipeline's auto_transcribe path degrades gracefully when
    whisper is not installed (transcribe() returns None → SubtitleSync(None)).
    """

    def test_pipeline_run_function_signature_accepts_auto_transcribe(self):
        """pipeline.run() must accept auto_transcribe, whisper_model, whisper_language."""
        import inspect
        from asvl.pipeline import run
        sig = inspect.signature(run)
        params = set(sig.parameters.keys())
        self.assertIn("auto_transcribe", params)
        self.assertIn("whisper_model", params)
        self.assertIn("whisper_language", params)

    def test_pipeline_auto_transcribe_defaults_to_true(self):
        """auto_transcribe should default to True."""
        import inspect
        from asvl.pipeline import run
        sig = inspect.signature(run)
        self.assertTrue(sig.parameters["auto_transcribe"].default is True)


if __name__ == "__main__":
    unittest.main()
