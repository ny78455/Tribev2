"""
asvl/asvl/features/__init__.py
Feature extractor subpackage.
"""
from .motion import compute_motion_score
from .scene import compute_scene_score
from .edges import compute_edge_diff
from .blur import is_blurred, is_black_frame
from .novelty import compute_novelty

__all__ = [
    "compute_motion_score",
    "compute_scene_score",
    "compute_edge_diff",
    "is_blurred",
    "is_black_frame",
    "compute_novelty",
]
