"""
aese/adapters/vlm_router.py
Central VLM routing layer.

Holds the user-selected VLM backend and exposes a unified API that
scene_label.py, summary.py, and any other consumer can import without
knowing which backend is active.

Supported backends
------------------
  fastvlm   — apple/FastVLM-0.5B  (lightweight, fast, CUDA-optional)
  gemma4    — google/gemma-4-E2B-it (high quality, needs ≥ 24 GB VRAM)
  yunet     — No generative VLM; character detection uses YuNet/OpenCV only.
              describe_scene / caption_event always return fallback values.

Usage
-----
  # Set once at startup (e.g. from cli.py after parsing --vlm):
  from aese.adapters.vlm_router import set_backend, describe_scene, caption_event

  set_backend("gemma4")          # or "fastvlm" or "yunet"
  label = describe_scene(image)
  caption = caption_event(image, scene, action, dialogue)

  # Read the current backend name at any time:
  from aese.adapters.vlm_router import get_backend
  print(get_backend())           # "fastvlm" | "gemma4" | "yunet"

Thread safety
-------------
  set_backend() is expected to be called ONCE before any parallel work begins.
  After that, all public functions are read-only with respect to _backend and
  are safe to call from multiple threads.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default to fastvlm (existing behaviour preserved)
_VALID_BACKENDS = ("fastvlm", "gemma4", "yunet")
_backend: str = "fastvlm"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def set_backend(name: str) -> None:
    """
    Set the active VLM backend for this process.

    Args:
        name: One of "fastvlm", "gemma4", "yunet".

    Raises:
        ValueError: if name is not a recognised backend.
    """
    global _backend
    name = name.lower().strip()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown VLM backend {name!r}. "
            f"Choose one of: {', '.join(_VALID_BACKENDS)}"
        )
    _backend = name
    logger.info("AESE VLM router: backend set to %r", _backend)


def get_backend() -> str:
    """Return the currently active backend name."""
    return _backend


# ---------------------------------------------------------------------------
# Internal: retrieve the chosen adapter module
# ---------------------------------------------------------------------------

def _adapter():
    """
    Import and return the active adapter module.

    Returns the fastvlm or gemma4 module object, or None when the backend
    is "yunet" (no generative VLM).
    """
    if _backend == "gemma4":
        from . import gemma4
        return gemma4
    elif _backend == "fastvlm":
        from . import fastvlm
        return fastvlm
    else:  # yunet -- no VLM
        return None


# ---------------------------------------------------------------------------
# Public VLM API (drop-in for fastvlm.py / gemma4.py consumers)
# ---------------------------------------------------------------------------

def vlm_available() -> bool:
    """
    Return True if the active backend has a loaded VLM model.

    Always returns False for the "yunet" backend.
    """
    mod = _adapter()
    if mod is None:
        return False
    try:
        # Both adapters expose _ensure_loaded(); call it to trigger lazy load.
        return bool(mod._ensure_loaded())
    except Exception:
        return False


def describe_scene(image: np.ndarray) -> str:
    """
    Classify the scene in *image* using the active VLM backend.

    Returns "unknown" when the backend is "yunet" or the model fails.
    """
    mod = _adapter()
    if mod is None:
        return "unknown"
    try:
        return mod.describe_scene(image)
    except Exception as exc:
        logger.debug("AESE VLM router: describe_scene error (%s): %s", _backend, exc)
        return "unknown"


def caption_event(
    image: np.ndarray,
    scene_label: str,
    action_label: str,
    dialogue_text: Optional[str],
) -> str:
    """
    Generate a one-sentence event caption using the active VLM backend.

    Returns "" when the backend is "yunet" or the model fails (callers fall
    back to their template summary).
    """
    mod = _adapter()
    if mod is None:
        return ""
    try:
        return mod.caption_event(image, scene_label, action_label, dialogue_text)
    except Exception as exc:
        logger.debug("AESE VLM router: caption_event error (%s): %s", _backend, exc)
        return ""


def ask(
    image_rgb: Optional[np.ndarray],
    prompt: str,
    max_new_tokens: int = 60,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Low-level single-turn prompt through the active VLM.

    Args:
        image_rgb:      HxWx3 RGB numpy array, or None for text-only (gemma4 only).
        prompt:         User-turn text prompt.
        max_new_tokens: Maximum tokens to generate.
        system_prompt:  Optional system instruction. Gemma-4 places this in a
                        dedicated system role; FastVLM prepends it to the user turn.

    Returns:
        Decoded response string, or "" when unavailable / on failure.
    """
    mod = _adapter()
    if mod is None:
        return ""
    try:
        return mod._ask(image_rgb, prompt, max_new_tokens, system_prompt=system_prompt)
    except Exception as exc:
        logger.warning("AESE VLM router: _ask error (%s): %s", _backend, exc)
        return ""


def get_active_detector_mode() -> str:
    """
    Return a short string describing the active VLM/detector state.

    Examples: "fastvlm", "gemma4", "yunet", "unavailable".
    """
    if _backend == "yunet":
        return "yunet"
    mod = _adapter()
    if mod is None:
        return "unavailable"
    try:
        return mod.get_active_detector_mode()
    except Exception:
        return "unavailable"
