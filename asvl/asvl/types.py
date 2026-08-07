"""
asvl/types.py
Core data contracts for the Adaptive Streaming Video Loader.
All dataclasses are defined exactly per the ASVL engineering contract §4.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AudioFeatures:
    energy: float          # 0-1, RMS normalized
    mfcc: np.ndarray       # shape (13,)
    spectral_flux: float
    speech_prob: float     # 0-1
    music_prob: float      # 0-1
    silence: bool


@dataclass
class FrameFeatures:
    motion_score: float    # 0-1, optical flow magnitude
    scene_score: float     # 0-1, histogram + SSIM combined
    scene_change: bool
    edge_diff: float
    is_blurred: bool
    is_black_frame: bool
    novelty_score: float   # 0-1


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: float
    image: np.ndarray            # HxWx3 RGB
    fps_used: float
    motion_score: float
    scene_change: bool
    audio_energy: float
    novelty_score: float
    decision_reason: str
    subtitle_text: Optional[str] = None


@dataclass
class ASVLConfig:
    adaptive: bool = True
    minimum_fps: float = 0.5
    maximum_fps: float = 10.0
    motion_threshold: float = 0.4
    scene_threshold: float = 0.7
    audio_threshold: float = 0.6
    novelty_threshold: float = 0.5
    buffer_seconds: float = 10.0
    weights: dict = field(default_factory=lambda: {
        "motion": 0.3,
        "scene": 0.25,
        "audio": 0.2,
        "subtitle": 0.1,
        "novelty": 0.15,
    })
