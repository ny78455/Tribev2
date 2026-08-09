"""
aese/adapters/embedding.py
Multimodal embedding adapter — produces a single embedding vector from a frame image
and optional subtitle text.

REAL IMPLEMENTATION:
  - Uses OpenCLIP (ViT-B/32, frozen, off-the-shelf) for both image and text encoding.
  - Image embedding: shape (512,) for ViT-B/32.
  - Text embedding: shape (512,) — only computed when subtitle_text is non-empty.
  - Fusion: "concat" (default) → (1024,); "mean" → (512,).
  - If CLIP model download fails or open-clip-torch is unavailable, falls back to
    perceptual hash + color histogram concatenation (clearly marked STUB below).

DECISIONS.md reference: §2 (CLIP or hash fallback).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLIP model singleton — loaded once, reused for all frames
# ---------------------------------------------------------------------------
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_available = False
_clip_load_attempted = False  # guard: only try once, even on failure
_clip_dim = 512  # ViT-B/32 embedding dimension

_STUB_HASH_DIM = 64   # pHash bits
_STUB_HIST_BINS = 32  # per-channel histogram bins × 3 channels = 96
_STUB_DIM = _STUB_HASH_DIM + _STUB_HIST_BINS * 3  # 160

# Effective embedding dimension — set on first successful load
EMBEDDING_DIM: int = 1024  # default assuming CLIP concat mode; updated at load time


def _try_load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai") -> bool:
    """Attempt to load OpenCLIP model. Returns True on success."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_available, EMBEDDING_DIM
    try:
        import open_clip  # type: ignore
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer
        _clip_available = True
        _clip_dim = model.visual.output_dim  # typically 512 for ViT-B/32
        EMBEDDING_DIM = _clip_dim * 2  # concat fusion doubles the dimension
        logger.info(
            "AESE: CLIP model '%s' (%s) loaded. Image dim=%d, concat dim=%d",
            model_name, pretrained, _clip_dim, EMBEDDING_DIM,
        )
        return True
    except Exception as exc:
        logger.warning(
            "AESE: CLIP load failed (%s) — falling back to perceptual hash + histogram stub. "
            "See DECISIONS.md §2.",
            exc,
        )
        # STUB: CLIP unavailable — use hash+histogram fallback
        EMBEDDING_DIM = _STUB_DIM * 2  # match concat dimension for consistency
        return False


def _init_clip(model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
    """Initialize CLIP if not already done. Only attempts load once."""
    global _clip_available, _clip_load_attempted
    if _clip_load_attempted:
        return
    _clip_load_attempted = True
    _clip_available = _try_load_clip(model_name, pretrained)


# ---------------------------------------------------------------------------
# STUB fallback: perceptual hash + color histogram
# Used when CLIP is unavailable.
# STUB: perceptual hash + color histogram — not a semantic embedding.
# ---------------------------------------------------------------------------
def _phash_vector(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    # STUB: perceptual hash — encodes coarse structural layout, not semantics.
    Returns a float32 vector of shape (hash_size*hash_size,) with values in {0.0, 1.0}.
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray.astype(np.float32), (hash_size, hash_size))
    mean = small.mean()
    return (small > mean).astype(np.float32).flatten()


def _color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    # STUB: normalized per-channel color histogram.
    Returns a float32 vector of shape (bins*3,).
    """
    import cv2
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    combined = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    norm = combined.sum()
    return combined / norm if norm > 0 else combined


def _stub_embedding(image: np.ndarray) -> np.ndarray:
    """
    # STUB: perceptual hash + histogram embedding.
    Used when CLIP model is unavailable.
    NOT a semantic embedding — cannot capture scene semantics or object meaning.
    Shape: (_STUB_HASH_DIM + _STUB_HIST_BINS*3,) = (160,).
    """
    ph = _phash_vector(image, hash_size=8)   # 64-D
    ch = _color_histogram(image, bins=32)     # 96-D
    return np.concatenate([ph, ch])            # 160-D


# ---------------------------------------------------------------------------
# Primary interface
# ---------------------------------------------------------------------------
def compute_embedding(
    image: np.ndarray,
    subtitle_text: Optional[str] = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    fusion: str = "concat",
) -> np.ndarray:
    """
    Compute a multimodal embedding from an image frame and optional subtitle text.

    Fusion modes:
      "concat" (default): [image_emb | text_emb] — doubles dimension
      "mean": (image_emb + text_emb) / 2 — keeps single-encoder dimension

    When subtitle_text is absent or empty, the text embedding is a zero vector
    (same shape as image embedding) — the image embedding carries the full signal.

    Returns:
        np.ndarray: float32 embedding vector. Shape depends on fusion and CLIP availability:
          - CLIP + concat: (1024,)  for ViT-B/32
          - CLIP + mean:   (512,)   for ViT-B/32
          - STUB + concat: (320,)
          - STUB + mean:   (160,)
    """
    _init_clip(model_name, pretrained)

    has_text = bool(subtitle_text and subtitle_text.strip())

    if _clip_available and _clip_model is not None:
        import torch

        device = next(_clip_model.parameters()).device

        # --- Image encoding ---
        try:
            import PIL.Image as PILImage
            pil_img = PILImage.fromarray(image)
            img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_emb = _clip_model.encode_image(img_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_vec = img_emb.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.debug("CLIP image encoding failed: %s — using zeros", exc)
            img_vec = np.zeros(_clip_dim, dtype=np.float32)

        # --- Text encoding ---
        if has_text:
            try:
                tokens = _clip_tokenizer([subtitle_text]).to(device)
                with torch.no_grad():
                    txt_emb = _clip_model.encode_text(tokens)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                txt_vec = txt_emb.squeeze(0).cpu().numpy().astype(np.float32)
            except Exception as exc:
                logger.debug("CLIP text encoding failed: %s — using zeros", exc)
                txt_vec = np.zeros(_clip_dim, dtype=np.float32)
        else:
            txt_vec = np.zeros(_clip_dim, dtype=np.float32)

        if fusion == "mean":
            if has_text:
                return ((img_vec + txt_vec) / 2.0).astype(np.float32)
            return img_vec
        else:  # concat (default)
            return np.concatenate([img_vec, txt_vec]).astype(np.float32)

    else:
        # STUB: CLIP unavailable — use perceptual hash + histogram
        img_stub = _stub_embedding(image)
        txt_stub = np.zeros_like(img_stub)  # text side is always zero in stub mode
        if fusion == "mean":
            return img_stub
        else:
            return np.concatenate([img_stub, txt_stub]).astype(np.float32)
