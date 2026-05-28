"""
Abstract base class for all inpainting backends.

Why an ABC here?
  - Backends (SD1.5, FLUX Fill, cloud) have wildly different implementations
    but the pipeline only needs three things: check availability, run inpaint,
    and know what model it's talking to.
  - Swapping backends becomes a one-line change in pipeline.py — no logic
    changes, just a different class injected into the same interface.
  - Type-safe: mypy/pyright can verify callers without knowing the backend.

Device priority: mps → cuda → cpu
  - MPS = Apple Silicon GPU (Metal Performance Shaders)
  - CUDA = NVIDIA GPU
  - CPU = universal fallback (slow but always available)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class InpaintingBackend(ABC):
    """
    Abstract interface for all inpainting backends.

    Subclasses must implement `inpaint`, `is_available`, and `model_id`.
    The constructor signature is intentionally minimal — subclasses may
    accept additional kwargs for backend-specific configuration.
    """

    def __init__(self, device: str = "mps") -> None:
        """
        Args:
            device: Compute device. One of "mps", "cuda", "cpu".
                    Subclasses resolve the actual device (e.g. fall back
                    from mps to cpu if torch.backends.mps.is_available()
                    returns False).
        """
        self.device = device

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        denoise: float,
        seed: int,
    ) -> Image.Image:
        """
        Run inpainting on the given image.

        The mask follows diffusers convention:
          - White (255) = region to inpaint (replace)
          - Black (0)   = region to keep (preserve)

        Args:
            image:           RGB PIL image, any size.
            mask:            Grayscale PIL image, same size as `image`.
            prompt:          Positive prompt describing desired fill.
            negative_prompt: Things to avoid in the generation.
            denoise:         Inpainting strength [0.0, 1.0].
                             Low (0.2-0.4) = subtle changes, preserves context.
                             High (0.7-1.0) = aggressive fill, creative freedom.
            seed:            RNG seed for reproducibility. Use -1 for random.

        Returns:
            RGB PIL image of the same dimensions as `image`.

        Raises:
            RuntimeError: If the backend is not available or inference fails.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this backend can actually run.

        For local model backends: verify model weights exist on disk.
        For cloud backends: verify API key is set and endpoint reachable.

        Does NOT trigger a model download — purely a readiness check.

        Returns:
            True if the backend is ready to inpaint.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Human-readable identifier for the model/backend.

        Examples:
          "runwayml/stable-diffusion-inpainting"
          "black-forest-labs/FLUX.1-Fill-dev"
          "replicate/stability-ai/stable-diffusion-inpainting"
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id!r}, device={self.device!r})"
