"""
aese/adapters/scene_label.py
Scene labeling adapter.

V3: Primary path routes through vlm_router so the active backend
(fastvlm / gemma4 / yunet) is respected without hard-coding fastvlm here.

Fallback chain (applied in order when image is unavailable or model fails):
  1. Active VLM backend via vlm_router (fastvlm / gemma4)
  2. CLIP zero-shot against the same label set -- if open_clip is available
  3. Color-temperature heuristic -- last resort

Falls back to "unknown" if all methods fail or if the image is None/black.
See DECISIONS.md S3.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fixed label set -- must stay in sync with fastvlm.py._SCENE_LABELS
# Exported as SCENE_LABELS (public) for tests and CLI banner.
# See Fix 3 for expanded vocabulary.
SCENE_LABELS = [
    "kitchen", "living room", "bedroom", "office", "hallway",
    "street", "village", "forest", "beach", "outdoor field",
    "vehicle interior", "rooftop", "restaurant", "stage/studio",
    "unknown",
]
# Backward-compatible private alias used internally
_SCENE_LABELS = SCENE_LABELS

# CLIP text features cached at first call (fallback path)
_clip_text_features = None
_clip_labels_loaded = False


def _clip_available() -> bool:
    """
    Return True if the CLIP model is loaded and available for scene classification.

    Does NOT trigger a load attempt -- this is a safe read of the current state.
    Used by cli.py for the SCENE CLASSIFICATION MODE startup banner.
    """
    try:
        from .embedding import _clip_available as _emb_clip_available
        return bool(_emb_clip_available)
    except Exception:
        return False


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

    Primary path: Active VLM backend via vlm_router (fastvlm / gemma4).
    Fallback 1:   CLIP zero-shot (if open_clip is available).
    Fallback 2:   Color-temperature heuristic (last resort).

    Args:
        image: HxWx3 RGB numpy array. Must not be None (caller filters None images).
               Black frames (image.max() < 5) short-circuit to "unknown".

    Returns:
        str: One of the labels in SCENE_LABELS. ALWAYS returns a str, never None.
             Returns "unknown" on any failure.
    """
    if image is None or image.max() < 5:
        return "unknown"

    # --- Path 1: Active VLM backend (fastvlm / gemma4 / yunet via router) ---
    try:
        from .vlm_router import describe_scene as _vlm_describe_scene, vlm_available
        if vlm_available():
            result = _vlm_describe_scene(image)
            if result and result != "unknown":
                return result
            # VLM returned "unknown" -- trust it and skip CLIP
            return "unknown"
    except Exception as exc:
        logger.debug("AESE scene_label: VLM path failed: %s -- trying CLIP", exc)

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
        logger.debug("AESE scene_label CLIP inference failed: %s -- using heuristic", exc)

    # --- Path 3: heuristic ---
    return _heuristic_scene_label(image)


def _heuristic_scene_label(image: np.ndarray) -> str:
    """
    Bare-minimum heuristic for scene label when VLM and CLIP are unavailable.
    Uses color temperature (blue channel ratio for sky/outdoor, warm tones for indoor).
    Not reliable -- only a last-resort fallback. Always returns a label in SCENE_LABELS.
    """
    try:
        h, w = image.shape[:2]
        top = image[:h // 3, :, :]
        mean_r = float(top[:, :, 0].mean())
        mean_b = float(top[:, :, 2].mean())
        if mean_b > mean_r + 15 and mean_b > 80:
            return "outdoor field"  # blue top third => sky visible
        brightness = float(image.mean())
        if brightness < 30:
            return "unknown"        # too dark to classify safely
        # Generic warm/neutral interior -- use "office" as the nearest non-specific label
        return "office"
    except Exception:
        return "unknown"
