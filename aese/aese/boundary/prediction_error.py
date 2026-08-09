"""
aese/boundary/prediction_error.py
Prediction Error Model — measures how surprising the current embedding is,
given the recent trajectory of embeddings.

V1: Linear extrapolation from the last 2-3 embeddings (or simple EMA).
NOT a trained transformer. This is the simplified version required by §5.5:

  "Build the simple version first: linear extrapolation from the last 2-3 embeddings
   or simple EMA. NOT a trained transformer — that's future work."

Extension point:
# TODO: replace with trained temporal transformer per Section 14.
  The interface (buffer → float) remains stable; only this module's internals change.
  A trained model would learn the typical trajectory of embeddings within a scene
  and produce a large error when an unexpected scene transition occurs.

See DECISIONS.md §9.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .embedding_change import embedding_distance


def predict_next_embedding(recent_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    V1: Predict the next embedding via linear extrapolation from recent history.

    Prediction rules:
      - If ≥ 2 embeddings available: predicted = last + (last - second_to_last)
        i.e. continue the linear trend from the last step.
      - If 1 embedding available: predicted = that embedding (no trend to extrapolate).
      - If 0 embeddings: return zero vector.

    # TODO: replace with trained temporal transformer per Section 14.

    Args:
        recent_embeddings: List of recent embedding vectors (oldest first), length ≥ 0.

    Returns:
        np.ndarray: Predicted embedding, same shape as input embeddings.
    """
    if len(recent_embeddings) == 0:
        return np.zeros(1, dtype=np.float32)  # fallback; shape updated by caller

    if len(recent_embeddings) == 1:
        return recent_embeddings[-1].copy()

    # Linear extrapolation: use last two points to project one step forward
    last = recent_embeddings[-1].astype(np.float64)
    second_last = recent_embeddings[-2].astype(np.float64)
    predicted = last + (last - second_last)

    # Normalize to unit sphere (CLIP embeddings are normalized)
    norm = np.linalg.norm(predicted)
    if norm > 1e-10:
        predicted = predicted / norm

    return predicted.astype(np.float32)


def compute_prediction_error(
    buffer_embeddings: List[np.ndarray],
    actual_embedding: np.ndarray,
    metric: str = "cosine",
) -> float:
    """
    Compute the prediction error: how different is the actual embedding from
    what we predicted based on recent history?

    High error (→ 1.0) indicates a surprising frame — a likely event boundary.
    Low error (→ 0.0) indicates the scene is continuing as expected.

    Args:
        buffer_embeddings: Recent embeddings from ContextBuffer (oldest first).
        actual_embedding: The actual current TemporalFeature embedding.
        metric: Distance metric for embedding_distance (default "cosine").

    Returns:
        float: Prediction error in [0, 1].
    """
    if not buffer_embeddings:
        return 0.0  # No history → neutral, not surprising

    predicted = predict_next_embedding(buffer_embeddings)

    # Handle shape mismatch (e.g. before CLIP is initialized)
    if predicted.shape != actual_embedding.shape:
        return 0.0

    return embedding_distance(predicted, actual_embedding, metric=metric)
