"""
aese/boundary/candidate_detector.py
Boundary Candidate Detector — §5.7, §16.

Every second, computes all boundary signals and fuses them into a score.
Implements the 2-second confidence hold to prevent boundary flapping:

  If fused_score crosses boundary_threshold but confidence is low (score near
  the margin), the detector holds the decision for up to 2 more seconds, collecting
  fresh signals before committing. This prevents spurious boundaries from
  momentary threshold crossings.

Non-functional requirements:
  - Never holds more than 2000ms before committing (§8)
  - No flapping on scores that oscillate around the threshold
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from ..context_buffer import ContextBuffer
from ..types import AESEConfig, BoundaryDecision, BoundarySignal, TemporalFeature
from .confidence import compute_confidence, is_high_confidence
from .embedding_change import embedding_distance
from .fusion import dominant_signal_name, fuse
from .prediction_error import compute_prediction_error
from .signals import (
    camera_signal,
    character_signal,
    dialogue_signal,
    emotion_signal,
    music_signal,
    scene_signal,
)

logger = logging.getLogger(__name__)

_MAX_HOLD_MS = 2000.0  # Non-functional requirement: max decision delay


class CandidateDetector:
    """
    Stateful boundary candidate detector.

    Usage:
        detector = CandidateDetector(config, context_buffer)
        # Each second:
        result = detector.update(curr_feature, prev_feature)
        if result is not None:
            # A boundary decision was made (is_boundary may be True or False)
    """

    def __init__(self, config: AESEConfig, buffer: ContextBuffer) -> None:
        self.config = config
        self.buffer = buffer
        # Hold state: track pending low-confidence candidates
        self._hold_start_ms: Optional[float] = None
        self._hold_scores: List[float] = []
        self._hold_signals: List[BoundarySignal] = []

    def update(
        self,
        curr: TemporalFeature,
        prev: Optional[TemporalFeature],
    ) -> BoundaryDecision:
        """
        Process the current second's TemporalFeature and return a BoundaryDecision.

        Args:
            curr: Current second's TemporalFeature.
            prev: Previous second's TemporalFeature, or None for the first second.

        Returns:
            BoundaryDecision with is_boundary, confidence, dominant_signal, fused_score.
        """
        if prev is None:
            # First second — no decision possible yet
            return BoundaryDecision(
                is_boundary=False,
                confidence=0.0,
                dominant_signal="none",
                fused_score=0.0,
            )

        # --- Compute all signals ---
        signals = self._compute_signals(curr, prev)
        fused = fuse(signals, self.config.weights)
        confidence = compute_confidence(
            fused, self.config.boundary_threshold, self.config.confidence_margin
        )
        dominant = dominant_signal_name(signals, self.config.weights)

        threshold = self.config.boundary_threshold
        margin = self.config.confidence_margin
        curr_ts = curr.timestamp_ms

        # --- Well above threshold — immediate high-confidence boundary ---
        if fused >= threshold + margin:
            self._clear_hold()
            logger.debug(
                "AESE boundary confirmed at ts=%.0f ms: score=%.3f confidence=%.3f signal=%s",
                curr_ts, fused, confidence, dominant,
            )
            return BoundaryDecision(
                is_boundary=True,
                confidence=confidence,
                dominant_signal=dominant,
                fused_score=fused,
            )

        # --- In the low-confidence zone: threshold - margin ≤ score < threshold + margin ---
        if fused >= threshold - margin:
            if self._hold_start_ms is None:
                # Start a new hold
                self._hold_start_ms = curr_ts
                self._hold_scores = [fused]
                self._hold_signals = [signals]
                logger.debug(
                    "AESE boundary hold started at ts=%.0f ms: score=%.3f",
                    curr_ts, fused,
                )
                return BoundaryDecision(
                    is_boundary=False,  # not yet committed
                    confidence=confidence,
                    dominant_signal=dominant,
                    fused_score=fused,
                )
            else:
                # Continue the hold — accumulate scores
                self._hold_scores.append(fused)
                self._hold_signals.append(signals)
                hold_duration = curr_ts - self._hold_start_ms

                # Commit after 2s of holding OR if we have ≥ 2 scores
                if hold_duration >= _MAX_HOLD_MS or len(self._hold_scores) >= 2:
                    avg_score = float(np.mean(self._hold_scores))
                    avg_confidence = compute_confidence(avg_score, threshold, margin)
                    # Re-evaluate the combined signal
                    all_signals_fused = [fuse(s, self.config.weights) for s in self._hold_signals]
                    final_score = max(all_signals_fused)  # take the peak
                    final_conf = compute_confidence(final_score, threshold, margin)
                    is_boundary = final_score >= threshold - margin  # lenient after hold

                    # Find the dominant signal from the peak window
                    peak_idx = int(np.argmax(all_signals_fused))
                    final_dominant = dominant_signal_name(
                        self._hold_signals[peak_idx], self.config.weights
                    )

                    self._clear_hold()
                    logger.debug(
                        "AESE hold committed at ts=%.0f ms: score=%.3f is_boundary=%s "
                        "(hold=%.0f ms)",
                        curr_ts, final_score, is_boundary, hold_duration,
                    )
                    return BoundaryDecision(
                        is_boundary=is_boundary,
                        confidence=final_conf,
                        dominant_signal=final_dominant,
                        fused_score=final_score,
                    )
                else:
                    # Still within hold window — keep waiting
                    return BoundaryDecision(
                        is_boundary=False,
                        confidence=confidence,
                        dominant_signal=dominant,
                        fused_score=fused,
                    )

        # --- Below threshold (and below low-confidence zone) ---
        # If we were in a hold and score dropped away, cancel the hold
        if self._hold_start_ms is not None:
            hold_duration = curr_ts - self._hold_start_ms
            if hold_duration >= _MAX_HOLD_MS:
                # 2s elapsed without confirmation — commit as non-boundary
                self._clear_hold()
            # else: keep holding (score may recover next second)

        return BoundaryDecision(
            is_boundary=False,
            confidence=confidence,
            dominant_signal=dominant,
            fused_score=fused,
        )

    def _compute_signals(
        self, curr: TemporalFeature, prev: TemporalFeature
    ) -> BoundarySignal:
        """Compute all boundary signals for the current second."""
        # Embedding distance: current vs previous
        emb_dist = embedding_distance(
            curr.multimodal_embedding, prev.multimodal_embedding, metric="cosine"
        )

        # Prediction error: current embedding vs. predicted from buffer history
        recent_embs = self.buffer.recent_embeddings(n=3)
        pred_error = compute_prediction_error(recent_embs, curr.multimodal_embedding)

        return BoundarySignal(
            scene=scene_signal(curr, prev),
            character=character_signal(curr, prev),
            dialogue=dialogue_signal(curr, prev),
            camera=camera_signal(curr),
            emotion=emotion_signal(curr, prev),  # always 0.0 — see DECISIONS.md §8
            music=music_signal(curr, prev),
            embedding_distance=emb_dist,
            prediction_error=pred_error,
        )

    def _clear_hold(self) -> None:
        """Reset hold state."""
        self._hold_start_ms = None
        self._hold_scores = []
        self._hold_signals = []
