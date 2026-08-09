"""
aese/event_embedding.py
Event Embedding — pools all TemporalFeature embeddings within an event into
a single representative vector.

V1: Temporal mean pooling — simplest correct implementation (§5.11).

# TODO: attention/transformer pooling per Section 19.
  A learned pooling mechanism would weight embeddings by their importance within
  the event (e.g. attending to scene changes within a complex event). For V1,
  uniform mean pooling is sufficient and requires no training.

Dimension note (§5.11):
  CLIP ViT-B/32 produces 512-D embeddings per modality.
  With "concat" fusion (default), event embeddings are 1024-D.
  With "mean" fusion, they are 512-D.
  If the CLIP stub is used, dimensions are 320-D (concat) or 160-D (mean).
  The actual dimension used is logged at pipeline startup — no silent padding.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .types import TemporalFeature

logger = logging.getLogger(__name__)


def pool_event_embedding(features: List[TemporalFeature]) -> np.ndarray:
    """
    V1: Temporal mean pooling of multimodal_embedding over all features in the event.

    # TODO: replace with attention/transformer pooling per Section 19.

    Args:
        features: List of TemporalFeatures within the event window (≥ 1).

    Returns:
        np.ndarray: Mean-pooled embedding vector, float32.
                    Shape matches the per-feature embedding dimension.
    """
    if not features:
        # Fallback — should not happen in practice; return zero vector
        from .adapters.embedding import EMBEDDING_DIM
        logger.warning("AESE event_embedding: no features to pool — returning zero vector")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embeddings = [tf.multimodal_embedding for tf in features]

    # Validate all embeddings have the same shape
    shapes = {e.shape for e in embeddings}
    if len(shapes) > 1:
        logger.warning(
            "AESE event_embedding: mixed embedding shapes %s — using first shape, truncating others",
            shapes,
        )
        target_dim = embeddings[0].shape[0]
        embeddings = [e[:target_dim] if len(e) >= target_dim else e for e in embeddings]

    stacked = np.stack(embeddings, axis=0).astype(np.float64)
    pooled = stacked.mean(axis=0).astype(np.float32)
    return pooled
