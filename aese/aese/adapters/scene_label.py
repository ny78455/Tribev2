"""
aese/adapters/scene_label.py
Scene labeling adapter.

V2: Primary path uses riddhimanrana/fastvlm-0.5b-captions (FastVLM) via the
fastvlm.py singleton. FastVLM generates a scene description and the result is
mapped to the fixed 12-label vocabulary so the TemporalFeature.scene_label
contract is unchanged.

Fallback chain (applied in order when image is unavailable or model fails):
  1. FastVLM (riddhimanrana/fastvlm-0.5b-captions) — primary, VLM-quality
  2. CLIP zero-shot against the same label set — if open_clip is available
  3. Color-temperature heuristic — last resort

Falls back to "unknown" if all methods fail or if the image is None/black.
See DECISIONS.md §3.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fixed label set — must stay in sync with fastvlm.py._SCENE_LABELS
_SCENE_LABELS = [
    "indoor",
    "outdoor",
    "vehicle interior",
    "street",
    "nature",
    "building exterior",
    "office",
    "restaurant",
    "kitchen",
    "bedroom",
    "nighttime",
    "unknown",
]

# CLIP text features cached at first call (fallback path)
_clip_text_features = None
_clip_labels_loaded = False


def _load_clip_text_features() -> bool:
    """Cache CLIP text encodings for all scene labels. Returns True on success."""
    global _clip_text_features, _clip_labels_loaded
    if _clip_labels_loaded:
        return _clip_text_features is not None

    try:
        from .embedding import _clip_model, _clip_tokenizer, _clip_available
        import torch

        if not _clip_available or _clip_model is None:
            _clip_labels_loaded = True
            return False

        device = next(_clip_model.parameters()).device
        prompts = [f"a photo of {lbl}" for lbl in _SCENE_LABELS]
        tokens = _clip_tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        _clip_text_features = feats.cpu().numpy().astype(np.float32)
        _clip_labels_loaded = True
        logger.debug("AESE scene_label: cached CLIP text features for %d labels", len(_SCENE_LABELS))
        return True
    except Exception as exc:
        logger.debug("AESE scene_label: CLIP text feature cache failed: %s", exc)
        _clip_labels_loaded = True
        return False


def label_scene(image: np.ndarray) -> str:
    """
    Classify the scene label of a single frame.

    Primary path: FastVLM (riddhimanrana/fastvlm-0.5b-captions).
    Fallback 1:   CLIP zero-shot (if open_clip is available).
    Fallback 2:   Color-temperature heuristic (last resort).

    Args:
        image: HxWx3 RGB numpy array. Must not be None (caller filters None images).
               Black frames (image.max() < 5) short-circuit to "unknown".

    Returns:
        str: One of the labels in _SCENE_LABELS. Returns "unknown" on any failure.
    """
    if image is None or image.max() < 5:
        return "unknown"

    # --- Path 1: FastVLM ---
    try:
        from .fastvlm import describe_scene, _fastvlm_available
        # Only attempt if already loaded successfully, or on first call
        result = describe_scene(image)
        if result and result != "unknown":
            return result
        if _fastvlm_available:
            # VLM returned "unknown" — trust it and skip CLIP
            return "unknown"
    except Exception as exc:
        logger.debug("AESE scene_label: FastVLM path failed: %s — trying CLIP", exc)

    # --- Path 2: CLIP zero-shot ---
    try:
        from .embedding import _clip_model, _clip_preprocess, _clip_available
        import torch

        if _clip_available and _clip_model is not None and _load_clip_text_features():
            device = next(_clip_model.parameters()).device
            import PIL.Image as PILImage
            pil_img = PILImage.fromarray(image)
            img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = _clip_model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_vec = img_feat.squeeze(0).cpu().numpy().astype(np.float32)

            sims = _clip_text_features @ img_vec
            best_idx = int(np.argmax(sims))
            return _SCENE_LABELS[best_idx]

    except Exception as exc:
        logger.debug("AESE scene_label CLIP inference failed: %s — using heuristic", exc)

    # --- Path 3: heuristic ---
    return _heuristic_scene_label(image)


def _heuristic_scene_label(image: np.ndarray) -> str:
    """
    # STUB: Bare-minimum heuristic for scene label when VLM and CLIP are unavailable.
    Uses color temperature (blue channel ratio for sky/outdoor, warm tones for indoor).
    Not reliable — only a last-resort fallback.
    """
    try:
        h, w = image.shape[:2]
        top = image[:h // 3, :, :]
        mean_r = float(top[:, :, 0].mean())
        mean_b = float(top[:, :, 2].mean())
        if mean_b > mean_r + 15 and mean_b > 80:
            return "outdoor"
        if mean_r > mean_b + 10:
            return "indoor"
        brightness = float(image.mean())
        if brightness < 30:
            return "nighttime"
        return "indoor"
    except Exception:
        return "unknown"
