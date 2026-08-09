"""
aese/types.py
Core data contracts for the Adaptive Event Segmentation Engine (Module 2).
All dataclasses are defined exactly per the AESE engineering contract §4.

WEIGHTS NOTE (§4 acceptance test):
  The source spec's weights sum to 1.05 (0.25+0.20+0.15+0.10+0.15+0.15+0.05 = 1.05).
  This is a bug in the source spec — flagged, renormalized (÷1.05), and logged in DECISIONS.md.
  Renormalized values used here sum exactly to 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# FramePacket — Mirror of Module 1's output contract (§1.1)
# Used here so AESE has a self-contained type definition for its input.
# ---------------------------------------------------------------------------
@dataclass
class FramePacket:
    """
    Mirror of ASVL Module 1's FramePacket.
    AESE only reads these fields — no image pixel data is required for offline
    (manifest-replay) mode. In live streaming mode, .image is populated.
    """
    frame_id: int
    timestamp_ms: float
    fps_used: float
    motion_score: float
    scene_change: bool
    audio_energy: float
    novelty_score: float
    decision_reason: str
    subtitle_text: Optional[str] = None
    image: Optional[np.ndarray] = None  # HxWx3 RGB — None in manifest-replay mode


# ---------------------------------------------------------------------------
# TemporalFeature — one aggregated per-second record (Feature Aggregator output)
# ---------------------------------------------------------------------------
@dataclass
class TemporalFeature:
    """
    One aggregated 'per-second' feature record, output of the Feature Aggregator.
    All categorical fields are derived via adapters (see adapters/*.py).
    Fields marked STUB use heuristics or placeholders; see DECISIONS.md.
    """
    timestamp_ms: float
    scene_label: str                     # STUB — zero-shot CLIP or heuristic; see adapters/scene_label.py
    character_count: int                 # STUB — face count only, no identity; see adapters/character_stub.py
    action_label: str                    # STUB — motion_score bucket; see adapters/action_stub.py
    dialogue_present: bool
    dialogue_text: Optional[str]
    camera_cue: Optional[str]            # "cut" | "fade" | "black" | None — derived from scene_change
    music_mood: str                      # STUB heuristic bucket; see adapters/music_mood.py
    multimodal_embedding: np.ndarray     # shape (D,) — CLIP or hash fallback
    motion_score: float
    novelty_score: float
    audio_energy: float = 0.0           # carried from FramePacket.audio_energy
    spectral_flux: float = 0.0          # estimated from audio_energy delta


# ---------------------------------------------------------------------------
# BoundarySignal — raw signal strengths before fusion (§5.3)
# ---------------------------------------------------------------------------
@dataclass
class BoundarySignal:
    scene: float             # [0,1]
    character: float         # [0,1]
    dialogue: float          # [0,1]
    camera: float            # [0,1]
    emotion: float           # [0,1] — always 0.0 (STUB, no emotion model); see DECISIONS.md
    music: float             # [0,1]
    embedding_distance: float  # [0,1]
    prediction_error: float    # [0,1]


# ---------------------------------------------------------------------------
# BoundaryDecision — output of candidate_detector + confidence (§5.7)
# ---------------------------------------------------------------------------
@dataclass
class BoundaryDecision:
    is_boundary: bool
    confidence: float        # [0,1]
    dominant_signal: str     # e.g. "scene", "embedding" — for boundary_reason
    fused_score: float       # [0,1]


# ---------------------------------------------------------------------------
# Event — one semantic segment of the film (§4 contract, §17–§23 construction)
# ---------------------------------------------------------------------------
@dataclass
class Event:
    event_id: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    event_embedding: np.ndarray          # pooled vector, shape (D,)
    importance: float                    # [0,1]
    confidence: float                    # [0,1], from boundary that closed this event
    summary: str                         # STUB: template-based; NOT LLM-generated; see README.md
    boundary_reason: str                 # dominant signal name
    event_type: str                      # Section 23 categories: Dialogue/Action/Transition/Scene
    key_frame: Optional[np.ndarray] = None
    characters: List[int] = field(default_factory=list)   # STUB: list of character counts seen
    location_label: Optional[str] = None


# ---------------------------------------------------------------------------
# AESEConfig — runtime configuration (§4, §6)
# ---------------------------------------------------------------------------
_RAW_SPEC_WEIGHTS = {
    "prediction_error": 0.25,
    "scene": 0.20,
    "dialogue": 0.15,
    "emotion": 0.10,
    "character": 0.15,
    "embedding": 0.15,
    "music": 0.05,
}
_RAW_SUM = sum(_RAW_SPEC_WEIGHTS.values())  # = 1.05 — bug in source spec
_RENORM_WEIGHTS = {k: round(v / _RAW_SUM, 6) for k, v in _RAW_SPEC_WEIGHTS.items()}


@dataclass
class AESEConfig:
    buffer_seconds: float = 45.0
    boundary_threshold: float = 0.75
    # Renormalized weights (source spec sums to 1.05 — see DECISIONS.md §1)
    weights: dict = field(default_factory=lambda: dict(_RENORM_WEIGHTS))
    minimum_event_duration_s: float = 5.0
    maximum_event_duration_s: float = 300.0
    merge_threshold: float = 0.80
    # Confidence hold: number of extra seconds to wait when score is near threshold
    confidence_hold_seconds: float = 2.0
    confidence_margin: float = 0.05
    # Embedding settings
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    embedding_fusion: str = "concat"  # "concat" | "mean"


def _assert_weights_sum(cfg: AESEConfig) -> None:
    """
    §4 acceptance test: assert weights sum to ~1.0 (after renormalization).
    Raises AssertionError if the renormalization above is broken.
    """
    total = sum(cfg.weights.values())
    assert abs(total - 1.0) < 1e-4, (
        f"AESEConfig weights sum to {total:.6f}, expected 1.0. "
        "Check DECISIONS.md §1 for renormalization details."
    )


# Run the check at import time on the default config
_assert_weights_sum(AESEConfig())
