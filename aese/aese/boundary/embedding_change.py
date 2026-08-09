"""
aese/boundary/embedding_change.py
Embedding distance calculator — measures semantic dissimilarity between two embeddings.

Used for:
  - Boundary signal G: per-second embedding distance between consecutive TemporalFeatures
  - Prediction error computation in prediction_error.py
  - Event merge decision in event_merge.py

Metrics:
  "cosine": 1 - cosine_similarity, result in [0, 2] → normalized to [0, 1] by dividing by 2
  "euclidean": Euclidean distance, normalized by empirical max (set from buffer statistics)
"""
from __future__ import annotations

from typing import Literal

import numpy as np


def embedding_distance(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    metric: Literal["cosine", "euclidean"] = "cosine",
    euclidean_max: float = 10.0,
) -> float:
    """
    Compute normalized distance between two embedding vectors.

    Args:
        emb_a: First embedding vector, shape (D,).
        emb_b: Second embedding vector, shape (D,).
        metric: "cosine" (default) or "euclidean".
        euclidean_max: Normalization constant for euclidean mode.
            Set empirically based on buffer statistics; default 10.0 is conservative.

    Returns:
        float: Distance in [0, 1]. 0.0 = identical, 1.0 = maximally different.

    Acceptance test:
        - Identical vectors → 0.0
        - Orthogonal vectors → 0.5 (cosine: (1 - 0) / 2 = 0.5)
        - Opposite vectors → 1.0 (cosine: (1 - (-1)) / 2 = 1.0)
    """
    if emb_a is None or emb_b is None:
        return 0.0

    a = emb_a.astype(np.float64)
    b = emb_b.astype(np.float64)

    if a.shape != b.shape or a.size == 0:
        return 0.0

    if metric == "cosine":
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0  # zero vector → treat as identical (neutral)
        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
        cosine_sim = max(-1.0, min(1.0, cosine_sim))  # numerical clamp
        # cosine distance: (1 - sim) is in [0, 2]; normalize to [0, 1]
        return float((1.0 - cosine_sim) / 2.0)

    else:  # euclidean
        dist = float(np.linalg.norm(a - b))
        return float(min(dist / euclidean_max, 1.0))
