"""
Photo editing backends package.

Each backend implements the InpaintingBackend abstract interface.
Swapping backends is a one-file change — add a new module here and
pass --backend <name> to pipeline.py.

Current backends:
  - sd15_inpaint: Stable Diffusion 1.5 inpainting (local, MPS/CUDA/CPU)

Planned backends (TODOs):
  - flux_fill:    FLUX Fill model (higher quality, more VRAM)
  - pulid:        PuLID face-consistent generation
  - cloud_api:    Delegated to cloud (Replicate, Stability AI, etc.)
"""

from .base import InpaintingBackend
from .sd15_inpaint import SD15InpaintBackend

__all__ = ["InpaintingBackend", "SD15InpaintBackend"]

# Registry: maps --backend CLI names to classes
BACKEND_REGISTRY: dict[str, type[InpaintingBackend]] = {
    "sd15_inpaint": SD15InpaintBackend,
}
