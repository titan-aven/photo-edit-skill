"""
Post-generation quality scoring for inpainting results.

Why score outputs?
  - Diffusion models are probabilistic — some runs produce artifacts
  - With multiple variants, we can automatically pick the best
  - A threshold-based pass/fail gives the pipeline a clear signal
    to retry or fall back to a safer denoising strength

Current metrics:
  - face_ssim: Structural similarity of protected faces vs original
               Catches the most common failure: face distortion

Planned metrics (TODOs):
  - background_consistency: SSIM/FID on unmasked region
    (Did the model accidentally change things it shouldn't have?)
  - seam_score: Edge sharpness at mask boundary
    (Visible seams = bad blending)
  - fid_patch: Fréchet Inception Distance on the inpainted patch
    (Measures realism of the filled region, not just preservation)
  - clip_alignment: CLIP similarity between prompt and inpainted region
    (Did the model actually do what we asked?)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)

# Default threshold for face preservation
# 0.75 = noticeable structural similarity maintained — a reasonable bar
# for natural photos with moderate compression. Lower if you're working
# with heavily compressed images where SSIM is noisy.
DEFAULT_FACE_SSIM_THRESHOLD = 0.75


@dataclass
class QualityScore:
    """
    Quality assessment result for a single inpainting variant.

    Attributes:
        face_ssim:  Mean SSIM across all protected face regions [0.0, 1.0].
                    1.0 = perfect preservation; < 0.75 = distortion warning.
        passed:     True if all thresholds were met.
        warnings:   Human-readable list of quality issues found.
        score:      Composite quality score used for ranking (higher = better).
                    Currently equals face_ssim; more metrics will feed into this.
    """

    face_ssim: float
    passed: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """
        Composite score for ranking variants.

        Currently just face_ssim, but structured as a property so we can
        later weight multiple metrics without changing call sites.
        """
        return self.face_ssim

    def to_dict(self) -> dict:
        return {
            "face_ssim": round(self.face_ssim, 4),
            "passed": self.passed,
            "warnings": self.warnings,
            "score": round(self.score, 4),
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"QualityScore({status} | face_ssim={self.face_ssim:.3f} "
            f"| warnings={self.warnings})"
        )


class QualityChecker:
    """
    Scores inpainting variants and decides pass/fail.

    Usage:
        checker = QualityChecker(face_ssim_threshold=0.75)
        score = checker.score(original, result, protected_faces)
        if score.passed:
            keep_variant()
    """

    def __init__(self, face_ssim_threshold: float = DEFAULT_FACE_SSIM_THRESHOLD) -> None:
        """
        Args:
            face_ssim_threshold: Minimum acceptable face SSIM [0.0, 1.0].
                                 Variants below this are flagged with warnings.
        """
        self.face_ssim_threshold = face_ssim_threshold

    def score(
        self,
        original: Image.Image,
        result: Image.Image,
        protected_faces: list,  # List[FaceBox] — avoid circular import
    ) -> QualityScore:
        """
        Compute quality score for an inpainting result.

        Args:
            original:        Source image before inpainting.
            result:          Inpainted output to evaluate.
            protected_faces: FaceBox list from FaceGuard.detect_faces()
                             (only faces OUTSIDE the removal zone).

        Returns:
            QualityScore with pass/fail decision and detailed metrics.
        """
        from .face_guard import FaceGuard

        warnings: list[str] = []

        # --- Metric 1: Face preservation SSIM ---
        guard = FaceGuard()
        face_ssim = guard.faces_intact(original, result, protected_faces)

        if face_ssim < self.face_ssim_threshold:
            warnings.append(
                f"face_distortion (ssim={face_ssim:.3f} < threshold={self.face_ssim_threshold:.3f})"
            )
            logger.warning(
                "Face distortion detected: SSIM=%.3f (threshold=%.3f)",
                face_ssim,
                self.face_ssim_threshold,
            )

        # TODO: Metric 2: Background consistency check
        # Check that unmasked regions weren't accidentally changed
        # background_ssim = _score_background(original, result, mask)

        # TODO: Metric 3: Seam quality
        # Check for hard edges at mask boundary
        # seam_score = _score_seams(result, mask)

        passed = len(warnings) == 0

        quality = QualityScore(
            face_ssim=face_ssim,
            passed=passed,
            warnings=warnings,
        )

        logger.debug("Quality check: %s", quality)
        return quality

    def rank_variants(
        self,
        variants: list[tuple[Image.Image, QualityScore]],
    ) -> list[tuple[Image.Image, QualityScore]]:
        """
        Sort variants by quality score, best first.

        Passing variants always rank above failing ones, even if a failing
        variant has a higher raw score. Within each tier, sorts by score desc.

        Args:
            variants: List of (image, score) tuples.

        Returns:
            Sorted list, best variant first.
        """
        # Sort: passing first, then by score descending
        return sorted(
            variants,
            key=lambda x: (not x[1].passed, -x[1].score),
        )
