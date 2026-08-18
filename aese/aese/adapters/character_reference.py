"""
aese/adapters/character_reference.py
Optional named-reference matching for AESE (Fix 5).

This module is the ONLY path in AESE where a real person name can appear
in character_labels. It requires the user to explicitly supply labeled
reference photos -- it never attempts automatic recognition of unlabeled
or public figures.

Hard guardrail (see README.md and DECISIONS.md):
  match_to_reference() returns a name ONLY if:
    1. The user supplied that name via --character-references, AND
    2. A detected face embedding is within distance_threshold of the
       corresponding reference embedding.
  If neither condition holds, returns None -- the Fix 4 anonymous label
  is used instead. Real names are NEVER guessed.

Usage (from cli.py):
    gallery = load_reference_gallery({"refs/john.jpg": "John", "refs/sarah.jpg": "Sarah"})
    name = match_to_reference(face_emb, gallery)
    label = name if name else clusterer.assign(face_emb)  # fallback to anonymous
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_reference_gallery(
    ref_paths: Dict[str, str],
    min_face_size: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Load a named reference gallery from user-supplied labeled photos.

    Args:
        ref_paths: Mapping of {image_path: person_name}, e.g.
                   {"/path/to/john.jpg": "John"}.
                   Supplied by the user via --character-references.

    Returns:
        Dict mapping person_name -> face_embedding (normalized float32 np.ndarray).
        Entries are skipped with a WARNING if:
          - The image file cannot be loaded.
          - No face is detected in the image.
          - Face embedding fails.

    This function ONLY assigns names that the user explicitly provided.
    It performs no automatic recognition of unlabeled photos.
    """
    if not ref_paths:
        return {}

    gallery: Dict[str, np.ndarray] = {}

    for path, name in ref_paths.items():
        try:
            import cv2
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                logger.warning(
                    "AESE character_reference: could not load image %r for reference '%s' -- skipping",
                    path, name,
                )
                continue
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.warning(
                "AESE character_reference: failed to load %r: %s -- skipping '%s'",
                path, exc, name,
            )
            continue

        # Detect face boxes and embed
        try:
            from .character_stub import detect_faces_with_boxes
            from .character_cluster import extract_face_embeddings

            boxes = detect_faces_with_boxes(image)
            embeddings = extract_face_embeddings(image, boxes)
        except Exception as exc:
            logger.warning(
                "AESE character_reference: face detection/embedding failed for %r: %s -- skipping '%s'",
                path, exc, name,
            )
            continue

        if not embeddings:
            logger.warning(
                "AESE character_reference: no face detected in reference photo %r "
                "for '%s' -- skipping. Check that the photo contains a clearly visible face.",
                path, name,
            )
            continue

        # Use the first (largest/highest-confidence) face embedding for the reference
        gallery[name] = embeddings[0]
        logger.info("AESE character_reference: enrolled '%s' from %r", name, path)

    return gallery


def match_to_reference(
    face_embedding: np.ndarray,
    gallery: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> Optional[str]:
    """
    Match a face embedding against a user-supplied reference gallery.

    Returns a real name ONLY if:
      - The gallery is non-empty (user supplied references), AND
      - The best-matching reference is within distance threshold.

    Otherwise returns None -- the caller must fall back to an anonymous label.
    This function NEVER guesses or invents a name.

    Args:
        face_embedding: Normalized float32 embedding of a detected face.
        gallery:        {name: embedding} dict from load_reference_gallery().
        threshold:      L2 distance threshold (default 0.5 for normalized embeddings).

    Returns:
        str (real name) if matched within threshold, None otherwise.
    """
    if not gallery:
        return None

    dists = {
        name: float(np.linalg.norm(face_embedding - emb))
        for name, emb in gallery.items()
    }
    best_name = min(dists, key=dists.get)
    best_dist = dists[best_name]

    if best_dist < threshold:
        logger.debug(
            "AESE character_reference: matched face -> '%s' (dist=%.3f < threshold=%.3f)",
            best_name, best_dist, threshold,
        )
        return best_name

    logger.debug(
        "AESE character_reference: no match (best='%s', dist=%.3f >= threshold=%.3f) -- anonymous",
        best_name, best_dist, threshold,
    )
    return None