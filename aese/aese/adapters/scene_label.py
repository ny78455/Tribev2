"""
aese/adapters/scene_label.py
Scene labeling adapter.

# STUB: Zero-shot CLIP classification against a fixed 12-label set.
NOT a real scene-graph model — only coarse indoor/outdoor/vehicle/etc distinctions.
Expected precision: ~60-70% on common movie scenes; not suitable for fine-grained
location understanding. See DECISIONS.md §3.

Falls back to "unknown" if CLIP is unavailable (manifest-replay with no real frames).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fixed label set — coarse enough to be reliable under zero-shot CLIP
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

# CLIP text features cached at first call
_clip_text_features = None
_clip_labels_loaded = False


def _load_clip_text_features() -> bool:
    """Cache CLIP text encodings for all scene labels. Returns True on success."""
    global _clip_text_features, _clip_labels_loaded
    if _clip_labels_loaded:
        return _clip_text_features is not None

    try:
        # Reuse the shared CLIP model from embedding.py
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
    # STUB: Zero-shot CLIP scene classification against _SCENE_LABELS.

    Args:
        image: HxWx3 RGB numpy array. May be a black placeholder in manifest-replay mode.

    Returns:
        str: One of the labels in _SCENE_LABELS. Returns "unknown" on any failure.
    """
    # Black frame detection — manifest-replay mode yields black placeholders
    if image is None or image.max() < 5:
        return "unknown"

    try:
        from .embedding import _clip_model, _clip_preprocess, _clip_available
        import torch

        if not _clip_available or _clip_model is None or not _load_clip_text_features():
            return _heuristic_scene_label(image)

        device = next(_clip_model.parameters()).device
        import PIL.Image as PILImage
        pil_img = PILImage.fromarray(image)
        img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = _clip_model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_vec = img_feat.squeeze(0).cpu().numpy().astype(np.float32)

        # Cosine similarities against text features
        sims = _clip_text_features @ img_vec  # shape (N,)
        best_idx = int(np.argmax(sims))
        return _SCENE_LABELS[best_idx]

    except Exception as exc:
        logger.debug("AESE scene_label CLIP inference failed: %s — using heuristic", exc)
        return _heuristic_scene_label(image)


def _heuristic_scene_label(image: np.ndarray) -> str:
    """
    # STUB: Bare-minimum heuristic for scene label when CLIP is unavailable.
    Uses color temperature (blue channel ratio for sky/outdoor, warm tones for indoor).
    Not reliable — only a last-resort fallback.
    """
    try:
        import cv2
        h, w = image.shape[:2]
        # Sample top third (sky region)
        top = image[:h // 3, :, :]
        mean_r = float(top[:, :, 0].mean())
        mean_b = float(top[:, :, 2].mean())
        mean_g = float(top[:, :, 1].mean())
        # Blue dominance → outdoor/sky
        if mean_b > mean_r + 15 and mean_b > 80:
            return "outdoor"
        # Warm tones dominate → indoor
        if mean_r > mean_b + 10:
            return "indoor"
        # Dark frame
        brightness = float(image.mean())
        if brightness < 30:
            return "nighttime"
        return "indoor"
    except Exception:
        return "unknown"
