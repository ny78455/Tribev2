"""
aese/adapters/fastvlm.py
Singleton wrapper for the riddhimanrana/fastvlm-0.5b-captions VLM.

Provides three helpers that replace the static stub methods used by
scene_label.py, character_stub.py, and event_constructor.py:

  describe_scene(image)       -> str   (one of _SCENE_LABELS or "unknown")
  count_people(image)         -> int   (≥0, same contract as count_characters)
  caption_event(image, ...)   -> str   (rich one-sentence event description)

All three:
  - Return the original stub fallback value if the model is not installed,
    not loadable, or the image is None / black (image_available=False guard
    is applied by callers).
  - Load the model ONCE on first call (lazy, thread-safe via a module lock).
  - Use float16 when a CUDA device is available, float32 otherwise.

Dependencies (added to requirements.txt):
  transformers>=4.52.0   (FastVlmForConditionalGeneration first appeared here)
  accelerate>=0.26.0     (needed by device_map="auto")
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "riddhimanrana/fastvlm-0.5b-captions"

# Module-level singletons — set on first successful load
_model = None
_processor = None
_fastvlm_available: bool | None = None   # None = not yet attempted
_load_lock = threading.Lock()

# Scene label vocabulary — must stay in sync with scene_label.py
_SCENE_LABELS = [
    "indoor", "outdoor", "vehicle interior", "street", "nature",
    "building exterior", "office", "restaurant", "kitchen", "bedroom",
    "nighttime", "unknown",
]


# ---------------------------------------------------------------------------
# Internal: load model once
# ---------------------------------------------------------------------------

def _ensure_loaded() -> bool:
    """
    Attempt to load FastVLM model + processor.
    Returns True if successfully loaded, False if unavailable.
    Thread-safe — safe to call from multiple aggregator calls concurrently.
    """
    global _model, _processor, _fastvlm_available

    if _fastvlm_available is not None:
        return _fastvlm_available

    with _load_lock:
        if _fastvlm_available is not None:   # double-checked locking
            return _fastvlm_available
        try:
            import torch
            from transformers import AutoProcessor, FastVlmForConditionalGeneration

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info(
                "AESE FastVLM: loading %s (dtype=%s, device_map=auto) ...",
                _MODEL_ID, dtype,
            )
            _processor = AutoProcessor.from_pretrained(
                _MODEL_ID, trust_remote_code=True
            )
            _model = FastVlmForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            _model.eval()
            _fastvlm_available = True
            logger.info("AESE FastVLM: model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "AESE FastVLM: load failed (%s) — "
                "scene_label, character_count, and event summary will use "
                "static fallbacks. Install transformers>=4.52 and accelerate "
                "to enable VLM-powered event generation.",
                exc,
            )
            _fastvlm_available = False

    return _fastvlm_available


def _ask(image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 60) -> str:
    """
    Run a single image + text prompt through FastVLM.
    Returns the model's response string, or "" on any failure.
    """
    if not _ensure_loaded():
        return ""

    try:
        import PIL.Image as PILImage
        import torch

        pil_img = PILImage.fromarray(image_rgb)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = _processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = _processor(
            text=text, images=pil_img, return_tensors="pt"
        ).to(_model.device)

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy — deterministic, fast
            )

        # Decode only the newly generated tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response = _processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return response

    except Exception as exc:
        logger.debug("AESE FastVLM: inference error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API — drop-in replacements for static stub methods
# ---------------------------------------------------------------------------

def describe_scene(image: np.ndarray) -> str:
    """
    Use FastVLM to classify the scene into one of the fixed _SCENE_LABELS.

    Prompt engineering: ask VLM to choose from the label vocabulary so the
    output stays compatible with the existing TemporalFeature.scene_label contract.

    Returns "unknown" on model failure or unavailability.
    """
    label_list = ", ".join(f'"{l}"' for l in _SCENE_LABELS if l != "unknown")
    prompt = (
        f"Look at this image and choose the single best matching scene label "
        f"from this list: {label_list}. "
        f"Reply with only the label, nothing else."
    )
    response = _ask(image, prompt, max_new_tokens=10)
    if not response:
        return "unknown"

    # Match response against the label vocabulary (case-insensitive)
    response_lower = response.lower().strip().strip('"').strip("'")
    for label in _SCENE_LABELS:
        if label in response_lower or response_lower in label:
            return label

    logger.debug("AESE FastVLM: scene label %r not in vocabulary — using 'unknown'", response)
    return "unknown"


def count_people(image: np.ndarray) -> int:
    """
    Use FastVLM to count the number of people visible in the frame.

    Returns an int ≥ 0. Returns 0 on model failure or unavailability.
    Same contract as count_characters() — no identity, just a count.
    """
    prompt = (
        "How many people are visible in this image? "
        "Reply with only a single integer (e.g. 0, 1, 2, 3)."
    )
    response = _ask(image, prompt, max_new_tokens=8)
    if not response:
        return 0

    # Extract first integer from the response
    match = re.search(r"\d+", response)
    if match:
        return int(match.group())

    logger.debug("AESE FastVLM: count_people response %r had no integer — returning 0", response)
    return 0


def caption_event(
    image: np.ndarray,
    scene_label: str,
    action_label: str,
    dialogue_text: Optional[str],
) -> str:
    """
    Use FastVLM to generate a rich one-sentence caption for an event.

    Falls back gracefully to None (caller uses template summary).

    Args:
        image:        Representative frame from the event.
        scene_label:  Aggregated scene label (used as context hint).
        action_label: Aggregated action label (used as context hint).
        dialogue_text: Most recent dialogue text in this event (or None).

    Returns:
        str: A one-sentence description of the event, or "" on failure.
    """
    ctx_parts = [f"scene: {scene_label}", f"action: {action_label}"]
    if dialogue_text:
        short_dialogue = dialogue_text[:80] + ("…" if len(dialogue_text) > 80 else "")
        ctx_parts.append(f'dialogue: "{short_dialogue}"')
    context = "; ".join(ctx_parts)

    prompt = (
        f"Describe what is happening in this movie scene in one concise sentence. "
        f"Context — {context}. "
        f"Be specific about actions and setting. Reply with one sentence only."
    )
    response = _ask(image, prompt, max_new_tokens=80)
    return response  # "" on failure — caller falls back to template
