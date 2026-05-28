"""
Face detection and protection for inpainting pipelines.

The core problem this solves:
  You want to remove a person from a group photo. The inpainting model
  doesn't know that the OTHER people in the photo should be untouched.
  Without protection, their faces often get subtly warped or blended.

Strategy:
  1. Detect all faces in the image (mediapipe, lightweight + MPS-compatible)
  2. Find which faces overlap with the removal mask (those we WANT to erase)
  3. Find faces NOT in the removal zone (those we WANT to preserve)
  4. Subtract protected face regions from the inpainting mask
  5. After generation, measure SSIM on protected regions to detect drift

Why mediapipe over face_recognition / dlib?
  - mediapipe runs on CPU but is fast (~10ms per image)
  - No CUDA requirement — works everywhere including Mac
  - Returns normalized bounding boxes (easy to convert to pixels)
  - face_recognition requires dlib which is painful to install on ARM

TODO: Consider using InsightFace for better accuracy on occluded faces
TODO: Add facial landmark tracking for finer-grained protection masks
TODO: Consider CLIP-based face identity verification instead of SSIM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """
    Bounding box for a detected face, in absolute pixel coordinates.

    Attributes:
        x:          Left edge (pixels from left of image)
        y:          Top edge (pixels from top of image)
        width:      Box width in pixels
        height:     Box height in pixels
        confidence: Detection confidence [0.0, 1.0]
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def x2(self) -> int:
        """Right edge pixel coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge pixel coordinate."""
        return self.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, x2, y2) suitable for PIL crop/draw operations."""
        return (self.x, self.y, self.x2, self.y2)

    def overlaps_mask(self, mask: Image.Image, threshold: float = 0.15) -> bool:
        """
        Check if this face overlaps significantly with a white region of a mask.

        We crop the mask to this face's bounding box and measure what fraction
        of pixels are white. If > threshold, the face is "in" the removal zone.

        Args:
            mask:      Grayscale mask (white=inpaint, black=keep).
            threshold: Fraction of white pixels needed to count as overlap.
                       Default 0.15 = 15% overlap triggers protection bypass.

        Returns:
            True if this face overlaps the mask region.
        """
        # Clamp box to image bounds
        iw, ih = mask.size
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(iw, self.x2)
        y2 = min(ih, self.y2)

        if x2 <= x1 or y2 <= y1:
            return False

        crop = mask.crop((x1, y1, x2, y2))
        arr = np.array(crop, dtype=np.float32) / 255.0
        white_fraction = float(arr.mean())
        return white_fraction > threshold


class FaceGuard:
    """
    Face detection and mask protection utility.

    Stateless — create one instance and reuse across images.
    The mediapipe detector is initialized lazily on first use.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Args:
            min_confidence: Minimum face detection confidence to include a face.
                            0.5 is a reasonable default; lower for occluded faces.
        """
        self.min_confidence = min_confidence
        self._detector = None  # Lazily initialized

    def _get_detector(self):
        """
        Initialize and cache the mediapipe face detector.

        Lazy init avoids importing mediapipe at module load time
        (it's slow to import and uses ~100 MB RAM).
        """
        if self._detector is not None:
            return self._detector

        try:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            self._detector = mp_face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range model (works beyond 2m)
                min_detection_confidence=self.min_confidence,
            )
            logger.debug("mediapipe face detector initialized")
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for face detection. "
                "Install it with: pip install mediapipe"
            ) from exc

        return self._detector

    def detect_faces(self, image: Image.Image) -> List[FaceBox]:
        """
        Detect all faces in the image.

        Args:
            image: RGB or RGBA PIL image.

        Returns:
            List of FaceBox objects, sorted by confidence (highest first).
            Empty list if no faces detected.
        """
        import mediapipe as mp

        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]

        detector = self._get_detector()
        results = detector.process(rgb)

        if not results.detections:
            logger.debug("No faces detected in image (%dx%d)", w, h)
            return []

        boxes = []
        for detection in results.detections:
            score = detection.score[0] if detection.score else 0.0
            bbox = detection.location_data.relative_bounding_box

            # Convert normalized [0,1] coordinates to absolute pixels
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Clamp to image bounds (mediapipe sometimes returns out-of-bounds boxes)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            boxes.append(FaceBox(x=x, y=y, width=bw, height=bh, confidence=score))

        boxes.sort(key=lambda b: b.confidence, reverse=True)
        logger.debug("Detected %d face(s)", len(boxes))
        return boxes

    def create_face_protection_mask(
        self,
        image: Image.Image,
        removal_mask: Image.Image,
        padding: int = 20,
    ) -> Image.Image:
        """
        Refine a removal mask to protect faces that should NOT be touched.

        Faces that overlap the removal zone are assumed to be the subject
        being removed — they're left in the mask. Faces that are OUTSIDE
        the removal zone are protected by subtracting them from the mask.

        Args:
            image:        Source RGB image.
            removal_mask: Original inpainting mask (white=inpaint region).
            padding:      Extra pixels to add around each protected face box.
                          Gives the model a buffer to blend cleanly at face edges.

        Returns:
            Refined mask where protected faces are blacked out (won't be inpainted).
        """
        from PIL import ImageDraw

        faces = self.detect_faces(image)
        if not faces:
            logger.debug("No faces found; returning mask unchanged")
            return removal_mask

        iw, ih = image.size
        protected = [f for f in faces if not f.overlaps_mask(removal_mask)]

        if not protected:
            logger.debug("All detected faces overlap removal zone; no protection needed")
            return removal_mask

        logger.info(
            "Protecting %d face(s) outside removal zone (out of %d total detected)",
            len(protected),
            len(faces),
        )

        # Start from the original mask and black-out protected face regions
        refined = removal_mask.copy()
        draw = ImageDraw.Draw(refined)

        for face in protected:
            # Expand box by padding, clamped to image bounds
            x1 = max(0, face.x - padding)
            y1 = max(0, face.y - padding)
            x2 = min(iw, face.x2 + padding)
            y2 = min(ih, face.y2 + padding)
            # Black = keep = don't inpaint this face
            draw.rectangle([x1, y1, x2, y2], fill=0)

        return refined

    def faces_intact(
        self,
        original: Image.Image,
        result: Image.Image,
        protected_boxes: List[FaceBox],
        padding: int = 10,
    ) -> float:
        """
        Measure how well protected faces were preserved in the inpainting result.

        For each protected face box, computes SSIM between the original crop
        and the result crop. Returns the mean SSIM across all protected faces.

        SSIM (Structural Similarity Index):
          - 1.0 = perfect preservation
          - ~0.9 = slight changes (acceptable)
          - <0.75 = noticeable face distortion (warning territory)
          - <0.5  = significant face damage

        Args:
            original:        Original source image.
            result:          Inpainted result image.
            protected_boxes: List of FaceBox objects that were protected.
            padding:         Margin to add around each face box for context.

        Returns:
            Mean SSIM score [0.0, 1.0]. Returns 1.0 if no protected boxes.
        """
        if not protected_boxes:
            return 1.0  # No faces to protect → vacuously perfect

        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError as exc:
            raise ImportError(
                "scikit-image is required for SSIM scoring. "
                "Install it with: pip install scikit-image"
            ) from exc

        iw, ih = original.size
        scores = []

        for face in protected_boxes:
            # Expand crop by padding for context, clamp to bounds
            x1 = max(0, face.x - padding)
            y1 = max(0, face.y - padding)
            x2 = min(iw, face.x2 + padding)
            y2 = min(ih, face.y2 + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            orig_crop = np.array(original.convert("RGB").crop((x1, y1, x2, y2)))
            result_crop = np.array(result.convert("RGB").crop((x1, y1, x2, y2)))

            if orig_crop.shape != result_crop.shape:
                # Shouldn't happen if both images are the same size, but guard anyway
                result_crop = np.array(
                    result.convert("RGB").crop((x1, y1, x2, y2)).resize(
                        (orig_crop.shape[1], orig_crop.shape[0]), Image.LANCZOS
                    )
                )

            score = ssim(
                orig_crop,
                result_crop,
                channel_axis=2,  # RGB channels
                data_range=255,
            )
            scores.append(float(score))
            logger.debug("Face SSIM at (%d,%d): %.3f", face.x, face.y, score)

        if not scores:
            return 1.0

        mean_ssim = float(np.mean(scores))
        logger.debug("Mean face SSIM across %d face(s): %.3f", len(scores), mean_ssim)
        return mean_ssim
