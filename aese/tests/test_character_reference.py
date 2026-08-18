"""
tests/test_character_reference.py
Tests for optional named-reference matching (Fix 5).

Guardrail: real names ONLY when the user has explicitly supplied them.
With no gallery -> always anonymous (Fix 4 behavior unchanged).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from aese.adapters.character_reference import match_to_reference, load_reference_gallery


def _vec(val: float, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = val
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# match_to_reference
# ---------------------------------------------------------------------------

def test_no_gallery_returns_none():
    """With no gallery supplied, always return None (anonymous label used instead)."""
    emb = _vec(1.0)
    result = match_to_reference(emb, {})
    assert result is None, f"Expected None for empty gallery, got {result!r}"


def test_matching_face_within_threshold_returns_name():
    """Embedding very close to a reference should return the real name."""
    ref_emb = _vec(1.0)
    gallery = {"John": ref_emb.copy()}
    # A slightly noisy version of the same embedding
    query = ref_emb + np.random.randn(16).astype(np.float32) * 0.001
    query /= np.linalg.norm(query) + 1e-9
    result = match_to_reference(query, gallery, threshold=0.5)
    assert result == "John", f"Expected 'John', got {result!r}"


def test_non_matching_face_returns_none():
    """Orthogonal embedding (far from reference) must return None."""
    ref_emb = _vec(1.0)         # points along axis 0
    gallery = {"John": ref_emb}
    query = np.zeros(16, dtype=np.float32)
    query[1] = 1.0              # orthogonal: distance = sqrt(2) >> threshold
    result = match_to_reference(query, gallery, threshold=0.5)
    assert result is None, (
        f"Expected None for non-matching face (should stay anonymous), got {result!r}"
    )


def test_two_references_picks_closest():
    """With two references, the closest one is returned."""
    john_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sarah_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    gallery = {"John": john_emb, "Sarah": sarah_emb}

    # Query closer to John
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    result = match_to_reference(query, gallery, threshold=1.0)
    assert result == "John", f"Expected 'John', got {result!r}"

    # Query closer to Sarah
    query2 = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    result2 = match_to_reference(query2, gallery, threshold=1.0)
    assert result2 == "Sarah", f"Expected 'Sarah', got {result2!r}"


def test_threshold_boundary_at_exactly_threshold():
    """Distance exactly equal to threshold should NOT match (strictly less than)."""
    ref_emb = np.array([1.0, 0.0], dtype=np.float32)
    gallery = {"John": ref_emb}
    # Distance = threshold exactly: should return None
    # dist = threshold means NOT matched (condition: dist < threshold)
    threshold = 0.5
    # Construct a vector at exactly threshold distance
    # ||q - r||^2 = threshold^2 -> q = [sqrt(1 - (t/2)^2), t/2]
    import math
    half_t = threshold / 2
    q = np.array([math.sqrt(1 - half_t**2), half_t], dtype=np.float32)
    dist = float(np.linalg.norm(q - ref_emb))
    # This test just verifies the strict inequality
    result = match_to_reference(q, gallery, threshold=dist)
    assert result is None, f"At exact threshold distance, expected None, got {result!r}"


# ---------------------------------------------------------------------------
# load_reference_gallery (error handling)
# ---------------------------------------------------------------------------

def test_load_empty_ref_paths_returns_empty():
    """Empty input -> empty gallery, no crash."""
    gallery = load_reference_gallery({})
    assert gallery == {}


def test_load_missing_file_skips_with_warning(caplog):
    """A missing image file should be skipped with a warning, not crash."""
    import logging
    with caplog.at_level(logging.WARNING, logger="aese.adapters.character_reference"):
        gallery = load_reference_gallery({"/nonexistent/path/photo.jpg": "Ghost"})
    assert "Ghost" not in gallery
    assert len(gallery) == 0