"""
aese/keyframe.py
Key frame selection from a list of TemporalFeatures.

Implements §5.10 — four named strategies behind a single interface.
Default: "lowest_blur" — cheapest and most robust since we don't have
per-frame images in all modes; uses motion_score as a blur proxy.

Strategies:
  "lowest_blur"     — frame with lowest motion_score (static = sharpest)
  "center"          — middle frame in the event window
  "highest_novelty" — frame with highest novelty_score
  "most_motion"     — frame with highest motion_score (most dynamic)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import TemporalFeature


def select_keyframe(
    features: List[TemporalFeature],
    strategy: str = "lowest_blur",
) -> Optional[np.ndarray]:
    """
    Select a representative 'key frame' embedding from an event's TemporalFeatures.

    In manifest-replay mode, no raw images are available — this function returns
    the representative *embedding vector* of the selected feature instead of
    a pixel image, since pixel data requires the live pipeline.

    Args:
        features: List of TemporalFeatures spanning the event.
        strategy: One of "lowest_blur" | "center" | "highest_novelty" | "most_motion".

    Returns:
        Optional[np.ndarray]: Embedding of the selected feature, or None if empty.
    """
    if not features:
        return None

    if strategy == "lowest_blur":
        # Lowest motion_score proxy for sharpness
        selected = min(features, key=lambda tf: tf.motion_score)

    elif strategy == "center":
        selected = features[len(features) // 2]

    elif strategy == "highest_novelty":
        selected = max(features, key=lambda tf: tf.novelty_score)

    elif strategy == "most_motion":
        selected = max(features, key=lambda tf: tf.motion_score)

    else:
        raise ValueError(f"Unknown keyframe strategy: {strategy!r}. "
                         "Choose from: lowest_blur, center, highest_novelty, most_motion")

    # Return the embedding as the keyframe proxy
    return selected.multimodal_embedding.copy()
