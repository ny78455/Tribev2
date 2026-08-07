"""
asvl/asvl/subtitles.py
SRT/VTT subtitle loading and timestamp sync.

Gracefully returns None / 0.0 if no subtitle file is provided — never raises.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SubtitleSync:
    """
    Loads and provides access to subtitle entries from .srt or .vtt files.

    All timestamp values are in milliseconds.
    Returns None / 0.0 gracefully when no subtitle file is loaded.
    """

    def __init__(self, subtitle_path: Optional[str] = None) -> None:
        """
        Args:
            subtitle_path: Path to .srt or .vtt file, or None to disable.
        """
        # List of (start_ms, end_ms, text) tuples, sorted by start_ms
        self._entries: list = []
        self._loaded = False

        if subtitle_path is None:
            return

        if not os.path.isfile(subtitle_path):
            logger.warning("Subtitle file not found: %s — subtitles disabled.", subtitle_path)
            return

        ext = os.path.splitext(subtitle_path)[1].lower()
        try:
            if ext == ".srt":
                self._load_srt(subtitle_path)
            elif ext in (".vtt", ".webvtt"):
                self._load_vtt(subtitle_path)
            else:
                logger.warning("Unsupported subtitle format '%s' — subtitles disabled.", ext)
                return
            self._loaded = True
            logger.info("Loaded %d subtitle entries from %s.", len(self._entries), subtitle_path)
        except Exception as exc:
            logger.error("Failed to load subtitles from %s: %s — subtitles disabled.", subtitle_path, exc)

    def _load_srt(self, path: str) -> None:
        import pysrt
        subs = pysrt.open(path, encoding="utf-8")
        for sub in subs:
            start_ms = (
                sub.start.hours * 3_600_000
                + sub.start.minutes * 60_000
                + sub.start.seconds * 1_000
                + sub.start.milliseconds
            )
            end_ms = (
                sub.end.hours * 3_600_000
                + sub.end.minutes * 60_000
                + sub.end.seconds * 1_000
                + sub.end.milliseconds
            )
            self._entries.append((float(start_ms), float(end_ms), sub.text))

    def _load_vtt(self, path: str) -> None:
        import webvtt
        for caption in webvtt.read(path):
            start_ms = self._vtt_time_to_ms(caption.start)
            end_ms = self._vtt_time_to_ms(caption.end)
            self._entries.append((float(start_ms), float(end_ms), caption.text))

    @staticmethod
    def _vtt_time_to_ms(time_str: str) -> float:
        """Convert 'HH:MM:SS.mmm' or 'MM:SS.mmm' to milliseconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            return 0.0
        seconds = float(s)
        return (int(h) * 3600 + int(m) * 60 + seconds) * 1000.0

    def get_subtitle_at(self, timestamp_ms: float) -> Optional[str]:
        """
        Return the subtitle text active at timestamp_ms, or None.

        Uses linear scan; acceptable for typical subtitle counts (<10k entries).
        For very large files, a binary search would be more efficient.
        """
        if not self._loaded:
            return None
        for start_ms, end_ms, text in self._entries:
            if start_ms <= timestamp_ms < end_ms:
                return text
        return None

    def subtitle_density(self, timestamp_ms: float, window_ms: float = 5000.0) -> float:
        """
        Compute what fraction of [timestamp_ms, timestamp_ms + window_ms]
        is covered by subtitle text.

        Args:
            timestamp_ms: Start of the query window (ms).
            window_ms: Window length (ms).

        Returns:
            float in [0, 1]: 0.0 = no subtitles, 1.0 = fully covered.
        """
        if not self._loaded or window_ms <= 0:
            return 0.0

        window_start = timestamp_ms
        window_end = timestamp_ms + window_ms
        covered_ms = 0.0

        for start_ms, end_ms, _ in self._entries:
            # Early exit — entries are sorted
            if start_ms >= window_end:
                break
            if end_ms <= window_start:
                continue
            overlap_start = max(start_ms, window_start)
            overlap_end = min(end_ms, window_end)
            if overlap_end > overlap_start:
                covered_ms += overlap_end - overlap_start

        return float(min(covered_ms / window_ms, 1.0))
