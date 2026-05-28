"""
Photo editing processors package.

Processors are stateless utilities that slot into the pipeline around the backend:
  - face_guard:    Detect faces, protect innocent bystanders from inpainting damage
  - masking:       Mask creation, loading, morphological operations
  - quality_check: Score outputs, flag distortions, rank variants

Design principle: processors don't know about backends and backends don't know
about processors. The pipeline.py orchestrator wires them together.
"""

from .face_guard import FaceBox, FaceGuard
from .masking import (
    combine_masks,
    create_ellipse_mask,
    dilate_mask,
    invert_mask,
    load_mask,
)
from .quality_check import QualityChecker, QualityScore

__all__ = [
    "FaceBox",
    "FaceGuard",
    "load_mask",
    "create_ellipse_mask",
    "dilate_mask",
    "invert_mask",
    "combine_masks",
    "QualityChecker",
    "QualityScore",
]
