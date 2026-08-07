"""
asvl/asvl/__init__.py
Package initializer for the asvl library.
"""
from .types import FramePacket, AudioFeatures, FrameFeatures, ASVLConfig

__all__ = ["FramePacket", "AudioFeatures", "FrameFeatures", "ASVLConfig"]


def run(*args, **kwargs):
    """Lazy import of pipeline.run to avoid circular imports at module load."""
    from .pipeline import run as _run
    return _run(*args, **kwargs)
