"""
tests/test_character_cluster.py
Tests for anonymous face clustering (Fix 4).

Covers:
  - Same embedding always maps to same label
  - Distinct embeddings create distinct labels
  - Consistent across interleaved re-appearances (ABBA sequence)
  - Overflow (>26 clusters) degrades gracefully
  - Centroid update does not spawn new clusters for near-identical embeddings
  - get_character_labels_for_event returns deduplicated sorted set
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from aese.adapters.character_cluster import (
    CharacterClusterer,
    get_character_labels_for_event,
    extract_face_embeddings,
)


def _vec(val: float, dim: int = 16) -> np.ndarray:
    """Unit vector in a fixed direction for testing."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = val
    return v / (np.linalg.norm(v) + 1e-9)


def _orthogonal_vecs(n: int, dim: int = 16) -> list:
    """Return n orthogonal unit vectors for distinct-cluster testing."""
    vecs = []
    for i in range(n):
        v = np.zeros(dim, dtype=np.float32)
        v[i % dim] = 1.0
        vecs.append(v)
    return vecs


# ---------------------------------------------------------------------------
# Basic assignment
# ---------------------------------------------------------------------------

def test_same_embedding_same_label():
    """Repeated identical embedding must always return 'Person A'."""
    c = CharacterClusterer()
    emb = _vec(1.0)
    labels = [c.assign(emb) for _ in range(5)]
    assert all(l == "Person A" for l in labels), f"Expected all 'Person A', got {labels}"


def test_two_distinct_embeddings_two_labels():
    """Two orthogonal embeddings must produce exactly two distinct labels."""
    c = CharacterClusterer()
    a, b = _orthogonal_vecs(2)
    la = c.assign(a)
    lb = c.assign(b)
    assert la != lb, "Distinct embeddings must produce distinct labels"
    assert la == "Person A"
    assert lb == "Person B"


def test_first_cluster_is_person_a():
    """The first embedded face must always become 'Person A'."""
    c = CharacterClusterer()
    label = c.assign(np.random.randn(16).astype(np.float32))
    assert label == "Person A"


# ---------------------------------------------------------------------------
# Consistency across re-appearances
# ---------------------------------------------------------------------------

def test_consistent_across_reordering():
    """
    ABBA sequence: A appears, then B, then B again, then A again.
    Both faces must keep their original labels throughout.
    """
    c = CharacterClusterer(distance_threshold=0.5)
    a, b = _orthogonal_vecs(2)

    la1 = c.assign(a)   # first appearance: Person A
    lb1 = c.assign(b)   # first appearance: Person B
    lb2 = c.assign(b)   # re-appearance: must still be Person B
    la2 = c.assign(a)   # re-appearance: must still be Person A

    assert la1 == la2 == "Person A", f"Expected 'Person A' throughout, got {la1}, {la2}"
    assert lb1 == lb2 == "Person B", f"Expected 'Person B' throughout, got {lb1}, {lb2}"


def test_five_alternating_faces_consistent():
    """Five distinct faces appearing in random order keep consistent labels."""
    c = CharacterClusterer(distance_threshold=0.5)
    vecs = _orthogonal_vecs(5)

    # First pass: register
    first_labels = [c.assign(v) for v in vecs]
    # Second pass: must return same labels
    second_labels = [c.assign(v) for v in vecs]

    assert first_labels == second_labels, (
        f"Labels changed across passes: {first_labels} vs {second_labels}"
    )


# ---------------------------------------------------------------------------
# Overflow / edge cases
# ---------------------------------------------------------------------------

def test_overflow_beyond_26_labels():
    """Creating more than 26 clusters must not crash -- use 'Person N' fallback."""
    c = CharacterClusterer(distance_threshold=0.01)  # tiny threshold -> always new cluster
    vecs = _orthogonal_vecs(30, dim=64)
    labels = [c.assign(v) for v in vecs]
    # All must be unique (each is a new cluster)
    assert len(set(labels)) == 30, "Expected 30 unique labels"
    # Labels beyond Z must not be single letters
    assert labels[26] == "Person 27", f"Expected 'Person 27' for idx 26, got {labels[26]}"


def test_centroid_update_no_new_cluster():
    """
    Near-identical embeddings (small noise) must converge to the same cluster,
    not spawn new ones.
    """
    c = CharacterClusterer(distance_threshold=0.5)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    labels = []
    for _ in range(10):
        noisy = base + np.random.randn(4).astype(np.float32) * 0.01
        noisy /= np.linalg.norm(noisy) + 1e-9
        labels.append(c.assign(noisy))
    assert all(l == "Person A" for l in labels), (
        f"Expected all 'Person A' for near-identical embeddings, got {labels}"
    )
    assert len(c.cluster_labels) == 1, (
        f"Expected 1 cluster for near-identical embeddings, got {len(c.cluster_labels)}"
    )


# ---------------------------------------------------------------------------
# get_character_labels_for_event
# ---------------------------------------------------------------------------

def test_get_labels_empty_returns_empty():
    """No embeddings -> empty label list."""
    c = CharacterClusterer()
    result = get_character_labels_for_event([], c)
    assert result == []


def test_get_labels_deduplicates():
    """Same face appearing multiple times in an event -> one label in output."""
    c = CharacterClusterer()
    emb = _vec(1.0)
    result = get_character_labels_for_event([[emb], [emb], [emb]], c)
    assert result == ["Person A"], f"Expected ['Person A'], got {result}"


def test_get_labels_two_people_sorted():
    """Two distinct faces -> two labels, sorted."""
    c = CharacterClusterer()
    a, b = _orthogonal_vecs(2)
    result = get_character_labels_for_event([[a], [b], [a, b]], c)
    assert result == ["Person A", "Person B"], f"Got {result}"


# ---------------------------------------------------------------------------
# extract_face_embeddings (graceful degradation)
# ---------------------------------------------------------------------------

def test_extract_empty_boxes_returns_empty():
    """No boxes -> no embeddings."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = extract_face_embeddings(img, [])
    assert result == []


def test_extract_none_image_returns_empty():
    """None image -> empty list, no crash."""
    result = extract_face_embeddings(None, [(0, 0, 32, 32)])
    assert result == []