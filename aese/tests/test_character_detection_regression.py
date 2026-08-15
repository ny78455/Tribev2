"""
tests/test_character_detection_regression.py
Regression guards for the AESE character detection pipeline.

Fix contract requirement:
  - POSITIVE CONTROL: a known real photograph containing a human face must
    produce count >= 1. This is the exact failure mode that was invisible before
    (a detector always returning 0 passed the old test suite, which only had
    negative and type-check tests).
  - NEGATIVE CONTROL: a black frame must produce count == 0. Ensures that the
    recall improvements introduced in V3 did not introduce false positives on
    blank input.

Fixture:
  skimage.data.astronaut() — a bundled 512x512 RGB photograph of an astronaut,
  one clear frontal face. Requires no network access; the image is embedded
  in the scikit-image package itself.
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module gracefully if scikit-image is not installed.
# This way the rest of the test suite still runs in minimal environments.
# ---------------------------------------------------------------------------
skimage_available = True
try:
    from skimage import data as skdata
except ImportError:  # pragma: no cover
    skimage_available = False

pytestmark = pytest.mark.skipif(
    not skimage_available,
    reason="scikit-image not installed — skipping face fixture tests",
)


from aese.adapters.character_stub import count_characters, get_effective_detector_chain


# ---------------------------------------------------------------------------
# Positive control — real face
# ---------------------------------------------------------------------------

def test_detector_finds_at_least_one_face_on_known_photo():
    """
    Regression guard for the '0 characters detected on a real photo' failure mode.

    Uses skimage.data.astronaut() — a bundled RGB photograph (no network fetch)
    with one clear frontal face — as a positive-control fixture.

    Definition of done: count >= 1, with the active detector chain reported in
    the assertion message for immediate diagnostics on failure.
    """
    img = skdata.astronaut()  # RGB uint8 ndarray, shape (512, 512, 3), one clear face
    chain = get_effective_detector_chain()
    count = count_characters(img)
    assert count >= 1, (
        f"Expected at least 1 face on a known real photograph, got {count}. "
        f"Active detector chain: {chain}. "
        "If this fails with chain='none', install OpenCV with haarcascades support. "
        "If it fails with a configured chain, the detector parameters need recalibration."
    )


# ---------------------------------------------------------------------------
# Negative control — blank frame
# ---------------------------------------------------------------------------

def test_black_frame_still_returns_zero():
    """
    Regression guard: recall improvements must NOT introduce false positives
    on entirely blank (black) input.

    A detector that flags faces on a black frame has a badly misconfigured
    threshold — this test ensures the V3 parameter loosening (scaleFactor,
    minNeighbors, minSize) stayed within safe bounds.
    """
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    count = count_characters(black)
    assert count == 0, (
        f"Expected 0 faces on a black frame, got {count}. "
        "The detection parameters are too aggressive — raises false positives on blank input."
    )


# ---------------------------------------------------------------------------
# Accessor sanity check
# ---------------------------------------------------------------------------

def test_get_effective_detector_chain_returns_nonempty_string():
    """
    get_effective_detector_chain() must always return a non-empty string.
    The CLI banner depends on this value being meaningful at startup.
    """
    chain = get_effective_detector_chain()
    assert isinstance(chain, str), f"Expected str, got {type(chain)}"
    assert len(chain) > 0, "Detector chain string must not be empty"
    assert chain in (
        "fastvlm",
        "dnn",
        "opencv_haar_frontal+profile",
        "opencv_haar_frontal",
        "none",
    ), f"Unexpected detector chain value: {chain!r}"


# ---------------------------------------------------------------------------
# None-image guard (was already tested in test_adapters.py — keep here too
# for completeness of this regression module)
# ---------------------------------------------------------------------------

def test_none_image_returns_zero():
    """count_characters(None) must always return 0 without crashing."""
    assert count_characters(None) == 0
