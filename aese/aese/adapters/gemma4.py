"""
aese/adapters/gemma4.py
Singleton adapter for google/gemma-4-E2B-it (Gemma 4 2B instruction-tuned).

Provides the same public surface as fastvlm.py so either adapter can be
used as a drop-in via vlm_router.py:

  describe_scene(image)       -> str   (one of SCENE_LABELS or "unknown")
  count_people(image)         -> int   (NOT USED -- see note below)
  caption_event(image, ...)   -> str   (rich one-sentence event description)

NOTE on character counting:
  count_people() is provided for API symmetry but MUST NOT be called by
  the character-counting hot path. Character counting is handled exclusively
  by the deterministic OpenCV chain in character_stub.py (see DECISIONS.md
  Section 16). This function returns 0 and logs a warning if called.

Dependencies (add to requirements.txt):
  transformers>=4.52.0
  accelerate>=0.26.0
  torch

Usage (via vlm_router, not directly):
  python cli.py --input manifest.jsonl --output events.jsonl --vlm gemma4
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "google/gemma-4-E2B-it"

# Module-level singletons
_model = None
_processor = None
_gemma4_available: bool | None = None   # None = not yet attempted
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal: load model once
# ---------------------------------------------------------------------------

def _ensure_loaded() -> bool:
    """
    Attempt to load Gemma-4 processor + model.
    Returns True if successfully loaded, False if unavailable.
    Thread-safe via double-checked locking.
    """
    global _model, _processor, _gemma4_available

    if _gemma4_available is not None:
        return _gemma4_available

    with _load_lock:
        if _gemma4_available is not None:
            return _gemma4_available
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForMultimodalLM

            dtype = "auto"
            logger.info(
                "AESE Gemma-4: loading %s (dtype=%s, device_map=auto) ...",
                _MODEL_ID, dtype,
            )
            _processor = AutoProcessor.from_pretrained(_MODEL_ID)
            _model = AutoModelForMultimodalLM.from_pretrained(
                _MODEL_ID,
                dtype=dtype,
                device_map="auto",
            )
            _model.eval()
            _gemma4_available = True
            logger.info("AESE Gemma-4: model loaded successfully.")
        except ImportError as exc:
            logger.warning(
                "AESE Gemma-4: load failed (missing dependency: %s). "
                "Install transformers>=4.52, accelerate, and torch to enable.",
                exc,
            )
            _gemma4_available = False
        except Exception as exc:
            logger.warning(
                "AESE Gemma-4: load failed (%s). "
                "Check model ID, disk space, and GPU memory.",
                exc,
            )
            _gemma4_available = False

    return _gemma4_available


def get_active_detector_mode() -> str:
    """Return 'gemma4' if the model loaded, 'unavailable' otherwise."""
    if _gemma4_available is None:
        _ensure_loaded()
    return "gemma4" if _gemma4_available else "unavailable"


# ---------------------------------------------------------------------------
# Internal: run a prompt through the model
# ---------------------------------------------------------------------------

def _ask(
    image_rgb: Optional[np.ndarray],
    prompt: str,
    max_new_tokens: int = 256,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Run a text-only or multimodal prompt through Gemma-4.

    Args:
        image_rgb:      HxWx3 RGB numpy array, or None for text-only inference.
        prompt:         User-turn text prompt.
        max_new_tokens: Maximum tokens to generate.
        system_prompt:  Optional system instruction placed before the user turn.

    Returns:
        Decoded response string, or "" on any failure.
    """
    if not _ensure_loaded():
        return ""

    try:
        import PIL.Image as PILImage
        import torch

        # Build the message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        if image_rgb is not None:
            pil_img = PILImage.fromarray(image_rgb)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text",  "text": prompt},
                    ],
                }
            )
        else:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            )

        # Use apply_chat_template to build tokenized inputs
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(_model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        raw = _processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        # parse_response is available on some processors -- skip if not present
        if hasattr(_processor, "parse_response"):
            try:
                raw = _processor.parse_response(raw) or raw
            except Exception:
                pass  # parse_response is optional; raw string is fine

        return raw

    except Exception as exc:
        logger.debug("AESE Gemma-4: inference error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API -- same surface as fastvlm.py
# ---------------------------------------------------------------------------

def describe_scene(image: np.ndarray) -> str:
    """
    Use Gemma-4 to classify the scene into one of the fixed SCENE_LABELS.

    Returns a label from SCENE_LABELS or "unknown" on failure.
    """
    from .scene_label import SCENE_LABELS
    label_list = ", ".join(f'"{l}"' for l in SCENE_LABELS if l != "unknown")
    prompt = (
        f"Look at this image and choose the single best matching scene label "
        f"from this list: {label_list}. "
        f"Reply with only the label, nothing else."
    )
    response = _ask(image, prompt, max_new_tokens=16)
    if not response:
        return "unknown"

    response_lower = response.lower().strip().strip('"').strip("'")
    for label in SCENE_LABELS:
        if label in response_lower or response_lower in label:
            return label

    logger.debug("AESE Gemma-4: scene label %r not in vocabulary -- using 'unknown'", response)
    return "unknown"


def count_people(image: np.ndarray) -> int:
    """
    API-symmetry stub. Character counting MUST use character_stub.py (DECISIONS.md §16).
    This function is NOT called by the pipeline and returns 0 if invoked directly.
    """
    logger.warning(
        "AESE Gemma-4: count_people() called -- this should not happen. "
        "Character counting must use the deterministic OpenCV chain in character_stub.py."
    )
    return 0


def caption_event(
    image: np.ndarray,
    scene_label: str,
    action_label: str,
    dialogue_text: Optional[str],
) -> str:
    """
    Use Gemma-4 to generate a rich one-sentence caption for an event.
    Returns "" on failure -- caller falls back to template summary.
    """
    ctx_parts = [f"scene: {scene_label}", f"action: {action_label}"]
    if dialogue_text:
        short_dialogue = dialogue_text[:80] + ("..." if len(dialogue_text) > 80 else "")
        ctx_parts.append(f'dialogue: "{short_dialogue}"')
    context = "; ".join(ctx_parts)

    prompt = (
        f"Describe what is happening in this movie scene in one concise sentence. "
        f"Context -- {context}. "
        f"Be specific about actions and setting. Reply with one sentence only."
    )
    return _ask(image, prompt, max_new_tokens=100)