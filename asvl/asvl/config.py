"""
asvl/asvl/config.py
Load and validate ASVL configuration from a YAML file.
Supports CLI overrides via keyword arguments.
"""
import logging
import os
from typing import Any, Dict, Optional

import yaml

from .types import ASVLConfig

logger = logging.getLogger(__name__)

_VALID_KEYS = {
    "adaptive": bool,
    "minimum_fps": float,
    "maximum_fps": float,
    "motion_threshold": float,
    "scene_threshold": float,
    "audio_threshold": float,
    "novelty_threshold": float,
    "buffer_seconds": float,
    "weights": dict,
}

_WEIGHT_KEYS = {"motion", "scene", "audio", "subtitle", "novelty"}


def _coerce(key: str, value: Any, expected_type: type) -> Any:
    """Attempt to coerce value to expected_type; raise ValueError on failure."""
    try:
        if expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        return expected_type(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Config key '{key}': cannot coerce {value!r} to {expected_type.__name__}") from exc


def _validate_ranges(cfg: ASVLConfig) -> None:
    """Validate that all config values are within acceptable ranges."""
    if cfg.minimum_fps <= 0:
        raise ValueError(f"minimum_fps must be > 0, got {cfg.minimum_fps}")
    if cfg.maximum_fps <= 0:
        raise ValueError(f"maximum_fps must be > 0, got {cfg.maximum_fps}")
    if cfg.minimum_fps > cfg.maximum_fps:
        raise ValueError(
            f"minimum_fps ({cfg.minimum_fps}) must be <= maximum_fps ({cfg.maximum_fps})"
        )
    for thr_name in ("motion_threshold", "scene_threshold", "audio_threshold", "novelty_threshold"):
        val = getattr(cfg, thr_name)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{thr_name} must be in [0, 1], got {val}")
    if cfg.buffer_seconds <= 0:
        raise ValueError(f"buffer_seconds must be > 0, got {cfg.buffer_seconds}")

    # Weights must sum to ≈1.0 and all be non-negative
    missing = _WEIGHT_KEYS - set(cfg.weights.keys())
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    weight_sum = sum(cfg.weights.values())
    if abs(weight_sum - 1.0) > 0.05:
        logger.warning(
            "Weights sum to %.3f (expected ~1.0) — scores will be proportionally scaled.",
            weight_sum,
        )
    for k, v in cfg.weights.items():
        if v < 0:
            raise ValueError(f"Weight '{k}' must be >= 0, got {v}")


def load_config(
    yaml_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ASVLConfig:
    """
    Load ASVLConfig from a YAML file with optional CLI overrides.

    Args:
        yaml_path: Path to a YAML config file. Uses defaults if None.
        overrides: Dict of key=value overrides (e.g. from CLI flags).

    Returns:
        Validated ASVLConfig instance.
    """
    raw: Dict[str, Any] = {}

    if yaml_path is not None:
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # Apply overrides
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    # Build ASVLConfig with defaults, coercing types
    cfg_kwargs: Dict[str, Any] = {}
    for key, expected_type in _VALID_KEYS.items():
        if key in raw:
            cfg_kwargs[key] = _coerce(key, raw[key], expected_type)

    cfg = ASVLConfig(**cfg_kwargs)
    _validate_ranges(cfg)
    return cfg
