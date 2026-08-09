"""
aese/config.py
Configuration loader for AESE — mirrors the ASVL config pattern.
Loads AESEConfig from a YAML file with optional overrides.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import yaml

from .types import AESEConfig, _assert_weights_sum

logger = logging.getLogger(__name__)


def load_config(
    yaml_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AESEConfig:
    """
    Load AESEConfig from a YAML file with optional CLI overrides.

    Args:
        yaml_path: Path to a YAML config file. Uses dataclass defaults if None.
        overrides: Dict of key=value pairs to override loaded config values.

    Returns:
        Validated AESEConfig instance.
    """
    raw: Dict[str, Any] = {}

    if yaml_path is not None:
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"AESE config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("AESE: loaded config from %s", yaml_path)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    # Build AESEConfig
    cfg_kwargs: Dict[str, Any] = {}

    if "buffer_seconds" in raw:
        cfg_kwargs["buffer_seconds"] = float(raw["buffer_seconds"])
    if "boundary_threshold" in raw:
        cfg_kwargs["boundary_threshold"] = float(raw["boundary_threshold"])
    if "minimum_event_duration" in raw:
        cfg_kwargs["minimum_event_duration_s"] = float(raw["minimum_event_duration"])
    if "maximum_event_duration" in raw:
        cfg_kwargs["maximum_event_duration_s"] = float(raw["maximum_event_duration"])
    if "merge_threshold" in raw:
        cfg_kwargs["merge_threshold"] = float(raw["merge_threshold"])
    if "clip_model" in raw:
        cfg_kwargs["clip_model"] = str(raw["clip_model"])
    if "clip_pretrained" in raw:
        cfg_kwargs["clip_pretrained"] = str(raw["clip_pretrained"])
    if "embedding_fusion" in raw:
        cfg_kwargs["embedding_fusion"] = str(raw["embedding_fusion"])

    if "weights" in raw:
        weights = dict(raw["weights"])
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "AESE: config weights sum to %.4f (expected 1.0) — renormalizing.", total
            )
            weights = {k: v / total for k, v in weights.items()}
        cfg_kwargs["weights"] = weights

    cfg = AESEConfig(**cfg_kwargs)

    # Validate
    _assert_weights_sum(cfg)
    if cfg.boundary_threshold <= 0.0 or cfg.boundary_threshold > 1.0:
        raise ValueError(f"boundary_threshold must be in (0, 1], got {cfg.boundary_threshold}")
    if cfg.minimum_event_duration_s <= 0:
        raise ValueError(f"minimum_event_duration_s must be > 0, got {cfg.minimum_event_duration_s}")
    if cfg.maximum_event_duration_s <= cfg.minimum_event_duration_s:
        raise ValueError(
            f"maximum_event_duration_s ({cfg.maximum_event_duration_s}) must be "
            f"> minimum_event_duration_s ({cfg.minimum_event_duration_s})"
        )

    return cfg
