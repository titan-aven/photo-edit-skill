"""
Mask utilities for inpainting pipelines.

Masks use the diffusers convention throughout this codebase:
  - White (255) = inpaint this region (replace/fill)
  - Black (0)   = keep this region (preserve exactly)

Why a separate masking module?
  - Mask manipulation is tedious NumPy/PIL work that clutters pipeline logic
  - These helpers are tested independently of any backend
  - The same operations are useful for both manual and auto-mask workflows

TODO: Add SAM (Segment Anything Model) integration for click-to-mask
TODO: Add GrabCut-based rough segmentation for quick masks without SAM
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def load_mask(path: str | "Path", target_size: tuple[int, int] | None = None) -> Image.Image:  # type: ignore[name-defined]
    """
    Load a mask image from disk and optionally resize to match the source photo.

    Handles both binary (0/255) and grayscale (soft-edge) masks.
    Resizes with LANCZOS to preserve soft edges; hard masks stay hard.

    Args:
        path:        Path to mask file (PNG, JPEG, etc.)
        target_size: (width, height) to resize to. None = keep original size.

    Returns:
        Grayscale PIL image (mode "L").

    Raises:
        FileNotFoundError: If the mask file doesn't exist.
        ValueError:        If the file can't be interpreted as an image.
    """
    from pathlib import Path

    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    try:
        mask = Image.open(mask_path).convert("L")
    except Exception as exc:
        raise ValueError(f"Cannot load mask from {mask_path}: {exc}") from exc

    if target_size is not None and mask.size != target_size:
        mask = mask.resize(target_size, Image.LANCZOS)

    return mask


def create_ellipse_mask(
    image_size: tuple[int, int],
    center: tuple[int, int],
    radius: int | tuple[int, int],
) -> Image.Image:
    """
    Create a simple ellipse mask — useful for quick manual masking during dev/debug.

    The ellipse region is WHITE (inpaint), everything else is BLACK (keep).

    Args:
        image_size: (width, height) of the output mask.
        center:     (x, y) pixel center of the ellipse.
        radius:     Single int for a circle, or (rx, ry) for an ellipse.

    Returns:
        Grayscale PIL image with white ellipse on black background.

    Example:
        mask = create_ellipse_mask((1024, 768), center=(400, 300), radius=80)
    """
    if isinstance(radius, int):
        rx, ry = radius, radius
    else:
        rx, ry = radius

    cx, cy = center
    mask = Image.new("L", image_size, 0)  # Start all-black (keep everything)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=255,  # White = inpaint this region
    )
    return mask


def dilate_mask(mask: Image.Image, pixels: int = 10) -> Image.Image:
    """
    Expand mask edges outward by `pixels`.

    Dilation prevents visible seams at mask boundaries by including a
    feathered border of context pixels in the inpainting region.

    Uses a Gaussian blur + threshold approach rather than true morphological
    dilation because PIL doesn't have MaxFilter but does have GaussianBlur,
    and the blur→threshold gives a soft feathered edge (actually nicer).

    Args:
        mask:   Input grayscale mask.
        pixels: Dilation amount in pixels.

    Returns:
        Dilated grayscale mask.
    """
    # MaxFilter would be the morphologically correct approach, but PIL's
    # RankFilter with rank=size*size-1 approximates it
    from PIL import ImageFilter

    # Use rank filter for true max-dilation
    size = pixels * 2 + 1  # kernel size must be odd
    # Rank = last element = max = morphological dilation
    dilated = mask.filter(ImageFilter.RankFilter(size=size, rank=size * size - 1))
    return dilated


def invert_mask(mask: Image.Image) -> Image.Image:
    """
    Invert a mask: white becomes black, black becomes white.

    Useful when you have a "keep" mask and need an "inpaint" mask,
    or when compositing multiple masks together.

    Args:
        mask: Grayscale mask to invert.

    Returns:
        Inverted grayscale mask.
    """
    arr = np.array(mask, dtype=np.uint8)
    return Image.fromarray(255 - arr, mode="L")


def combine_masks(
    mask_a: Image.Image,
    mask_b: Image.Image,
    mode: str = "union",
) -> Image.Image:
    """
    Combine two masks using set-like operations.

    Both masks must be the same size. If they differ, mask_b is resized
    to match mask_a.

    Modes:
      - "union":        White if EITHER mask is white (logical OR)
                        Use to merge: inpaint region A + inpaint region B
      - "intersection": White only if BOTH masks are white (logical AND)
                        Use to inpaint where two regions overlap
      - "subtract":     White where mask_a is white AND mask_b is black (A - B)
                        Use to punch holes in a mask (e.g. protect faces)

    Args:
        mask_a: First grayscale mask.
        mask_b: Second grayscale mask.
        mode:   Combination mode. One of "union", "intersection", "subtract".

    Returns:
        Combined grayscale mask.

    Raises:
        ValueError: If mode is unrecognized.
    """
    if mask_b.size != mask_a.size:
        mask_b = mask_b.resize(mask_a.size, Image.LANCZOS)

    a = np.array(mask_a, dtype=np.float32) / 255.0
    b = np.array(mask_b, dtype=np.float32) / 255.0

    if mode == "union":
        result = np.maximum(a, b)
    elif mode == "intersection":
        result = np.minimum(a, b)
    elif mode == "subtract":
        # Inpaint where A says yes, but zero out where B says yes
        result = np.clip(a - b, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mask combine mode: {mode!r}. Use 'union', 'intersection', or 'subtract'.")

    return Image.fromarray((result * 255).astype(np.uint8), mode="L")
