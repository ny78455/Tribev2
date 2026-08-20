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
