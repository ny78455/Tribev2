"""
aese/adapters/character_cluster.py
Anonymous-but-consistent face clustering for AESE (Fix 4).

Architectural contract:
  - Assigns detected faces to consistent labels ("Person A", "Person B", ...)
    across the ENTIRE video -- same face always gets the same label.
  - Labels are ANONYMOUS by default. Real names are NEVER assigned here.
    Real names only appear if the user supplies --character-references (Fix 5),
    which runs a separate match step in character_reference.py.
  - Streaming-compatible: online nearest-centroid assignment, no full offline re-fit.
  - If face detection or CLIP are unavailable, returns empty lists gracefully.

Character identity guardrail (see README.md and DECISIONS.md):
  AESE clusters faces into consistent anonymous labels within a single video.
  It does NOT perform real-world identity recognition and will NOT name real
  people, including public figures, unless a labeled reference is explicitly
  supplied per Fix 5.

See: character_reference.py for the opt-in named-matching layer.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of distinct clusters before we stop creating new ones.
# At label index 26 (beyond "Z") we use "Person 27" etc. -- still anonymous.
_MAX_CLUSTERS = 52  # A-Z then a-z


def _label_for_index(idx: int) -> str:
    """Generate a human-readable anonymous label for cluster index idx."""
    if idx < 26:
        return f"Person {chr(65 + idx)}"  # Person A .. Person Z
    return f"Person {idx + 1}"            # Person 27 .. (overflow fallback)


class CharacterClusterer:
    """
    Online nearest-centroid face clusterer.

    Maintains a running set of face-embedding clusters across the whole video.
    Each new detected face embedding is either assigned to an existing cluster
    (if within distance_threshold of a centroid) or creates a new one.

    Centroid update rule: exponential moving average (0.9 old + 0.1 new).
    This keeps centroids stable over time while allowing gradual drift
    (e.g., lighting changes across a long video).

    Args:
        distance_threshold: Cosine-like L2 distance threshold. Embeddings closer
            than this to a known centroid are assigned to that cluster.
            Default 0.6 works well for normalized 512-d CLIP embeddings.
    """

    def __init__(self, distance_threshold: float = 0.6) -> None:
        self.threshold = distance_threshold
        self.cluster_centroids: List[np.ndarray] = []
        self.cluster_labels: List[str] = []

    def assign(self, face_embedding: np.ndarray) -> str:
        """
        Assign a face embedding to an existing cluster or create a new one.

        Returns:
            str: Anonymous label, e.g. "Person A".
        """
        if not self.cluster_centroids:
            return self._new_cluster(face_embedding)

        dists = [
            float(np.linalg.norm(face_embedding - c))
            for c in self.cluster_centroids
        ]
        best_idx = int(np.argmin(dists))

        if dists[best_idx] < self.threshold:
            # Update centroid with exponential moving average
            self.cluster_centroids[best_idx] = (
                0.9 * self.cluster_centroids[best_idx] + 0.1 * face_embedding
            )
            return self.cluster_labels[best_idx]

        return self._new_cluster(face_embedding)

    def _new_cluster(self, face_embedding: np.ndarray) -> str:
        label = _label_for_index(len(self.cluster_labels))
        self.cluster_centroids.append(face_embedding.copy())
        self.cluster_labels.append(label)
        logger.debug("AESE character_cluster: new cluster %s (total=%d)", label, len(self.cluster_labels))
        return label

    def reset(self) -> None:
        """Reset all cluster state (used between test runs)."""
        self.cluster_centroids = []
        self.cluster_labels = []


# ---------------------------------------------------------------------------
# Face crop embedding
# ---------------------------------------------------------------------------

def _embed_face_crop(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Embed a face crop using CLIP image encoder.

    Returns a normalized float32 numpy array of shape (D,), or None if
    CLIP is unavailable or embedding fails.
    """
    try:
        from .embedding import _clip_model, _clip_preprocess, _clip_available
        import torch
        import PIL.Image as PILImage

        if not _clip_available or _clip_model is None:
            return None

        device = next(_clip_model.parameters()).device
        pil_img = PILImage.fromarray(crop)
        img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = _clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.squeeze(0).cpu().numpy().astype(np.float32)
        return vec
    except Exception as exc:
        logger.debug("AESE character_cluster: face crop embed failed: %s", exc)
        return None


def extract_face_embeddings(
    image: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    """
    Extract CLIP embeddings for each detected face crop.

    Args:
        image:      HxWx3 RGB numpy array.
        face_boxes: List of (x, y, w, h) bounding boxes from character_stub.

    Returns:
        List of normalized float32 embedding vectors. Empty list if CLIP is
        unavailable or no valid crops could be extracted.
    """
    if not face_boxes or image is None:
        return []

    h, w = image.shape[:2]
    embeddings = []
    for (bx, by, bw, bh) in face_boxes:
        # Clamp to image bounds
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = _embed_face_crop(crop)
        if emb is not None:
            embeddings.append(emb)

    return embeddings


# ---------------------------------------------------------------------------
# Per-event label resolution
# ---------------------------------------------------------------------------

def get_character_labels_for_event(
    face_embeddings_per_second: List[List[np.ndarray]],
    clusterer: CharacterClusterer,
) -> List[str]:
    """
    Assign all face embeddings in an event to cluster labels and return
    the deduplicated sorted set of labels seen during this event.

    Args:
        face_embeddings_per_second: List of per-second face embedding lists
            (from TemporalFeature.face_embeddings across the event window).
        clusterer: The global CharacterClusterer for this video run.

    Returns:
        Sorted list of unique anonymous labels, e.g. ["Person A", "Person B"].
        Empty list if no embeddings were available.
    """
    seen_labels = set()
    for second_embeddings in face_embeddings_per_second:
        for emb in second_embeddings:
            label = clusterer.assign(emb)
            seen_labels.add(label)
    return sorted(seen_labels)