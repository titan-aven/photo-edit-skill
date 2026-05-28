"""
Stable Diffusion 1.5 inpainting backend.

Uses the `diffusers` StableDiffusionInpaintPipeline with:
  - DPMSolverMultistepScheduler for fast, high-quality sampling
  - MPS (Apple Silicon) as primary target, CUDA secondary, CPU fallback
  - float16 precision on MPS/CUDA; float32 on CPU (float16 broken on CPU)

Model: runwayml/stable-diffusion-inpainting
  - Specifically fine-tuned for inpainting (not the regular SD1.5)
  - Downloads ~4 GB on first use from Hugging Face Hub
  - Cached at ~/.cache/huggingface/hub/ by default

Why DPMSolverMultistepScheduler?
  - 20-30 steps → quality comparable to DDIM at 50 steps
  - Saves ~40% inference time vs default PNDMScheduler
  - Well-tested with SD1.5 inpainting weights

TODO: Add FLUX Fill backend for better quality (needs more VRAM ~12 GB)
TODO: Add LoRA/ControlNet support for consistent person editing
TODO: Add attention slicing for low-VRAM devices
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .base import InpaintingBackend

logger = logging.getLogger(__name__)

_MODEL_ID = "runwayml/stable-diffusion-inpainting"

_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted face, deformed, ugly, artifacts, watermark, "
    "low quality, out of focus, overexposed, underexposed, noise, grain"
)


class SD15InpaintBackend(InpaintingBackend):
    """
    Stable Diffusion 1.5 inpainting via Hugging Face diffusers.

    Lazy-loads the pipeline on first `inpaint()` call to avoid
    eating ~4 GB of VRAM/RAM at import time.
    """

    def __init__(self, device: str = "mps") -> None:
        """
        Args:
            device: Preferred compute device ("mps", "cuda", "cpu").
                    Actual device is resolved at load time based on availability.
        """
        super().__init__(device)
        # Lazily loaded — don't import torch/diffusers until actually needed
        self._pipe = None
        self._resolved_device: str | None = None

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def is_available(self) -> bool:
        """
        Check if model weights are cached locally.

        We check the Hugging Face cache directory rather than attempting
        a network request — this way is_available() works offline.
        """
        try:
            from huggingface_hub import try_to_load_from_cache  # type: ignore
            from huggingface_hub.utils import EntryNotFoundError  # type: ignore

            # Check for the model index file as a proxy for a complete download
            result = try_to_load_from_cache(
                repo_id=_MODEL_ID,
                filename="model_index.json",
            )
            # Returns None or a sentinel if not cached; returns path str if found
            return result is not None and not isinstance(result, type(None))
        except Exception:
            # If huggingface_hub isn't importable, fall back to diffusers
            # presence check — if diffusers isn't installed, this backend
            # can't run regardless.
            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401
                return False  # diffusers present but model not confirmed cached
            except ImportError:
                return False

    def _resolve_device(self) -> str:
        """
        Pick the best available device based on the requested preference.

        Priority: requested device → mps → cuda → cpu
        We always fall back gracefully — better slow than broken.
        """
        import torch

        if self.device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available; falling back to CPU")
            return "cpu"
        elif self.device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available; falling back to CPU")
            return "cpu"
        else:
            return "cpu"

    def _load_pipeline(self) -> None:
        """
        Load the diffusers pipeline into memory.

        Called lazily on first inpaint() call. Subsequent calls skip loading.
        Downloading happens here if the model isn't cached — takes a while
        on first run (~4 GB). Progress is printed by huggingface_hub.
        """
        if self._pipe is not None:
            return  # Already loaded

        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

        self._resolved_device = self._resolve_device()

        # float16 halves VRAM usage on GPU; CPU doesn't support float16 well
        dtype = torch.float32 if self._resolved_device == "cpu" else torch.float16

        logger.info(
            "Loading %s on %s (dtype=%s)…",
            _MODEL_ID,
            self._resolved_device,
            dtype,
        )

        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=dtype,
            # safety_checker=None disables NSFW filter — remove if needed
            # safety_checker=None,
        )

        # Swap to the faster DPM solver
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )

        self._pipe = self._pipe.to(self._resolved_device)

        # Enable memory-efficient attention on MPS if xformers isn't available
        # (xformers doesn't support MPS yet as of mid-2024)
        if self._resolved_device != "mps":
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                logger.debug("xformers memory-efficient attention enabled")
            except Exception:
                logger.debug("xformers not available; using default attention")

        logger.info("Pipeline ready on %s", self._resolved_device)

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        denoise: float = 0.25,
        seed: int = -1,
    ) -> Image.Image:
        """
        Run SD1.5 inpainting.

        Args:
            image:           RGB source image.
            mask:            Grayscale mask (white=inpaint, black=keep).
            prompt:          What to fill the masked area with.
                             Empty string works for seamless background fill.
            negative_prompt: What to avoid; defaults to quality-preservation prompt.
            denoise:         Strength [0.0, 1.0]. 0.25 = conservative, 0.75 = creative.
            seed:            RNG seed. -1 = random (different every run).

        Returns:
            Inpainted PIL image, same size as input.

        Raises:
            RuntimeError: If inference fails.
            ImportError:  If diffusers/torch aren't installed.
        """
        import torch

        self._load_pipeline()
        assert self._pipe is not None  # mypy
        assert self._resolved_device is not None

        # Fewer steps for low denoise (quick, conservative); more for high denoise
        num_steps = 30 if denoise < 0.4 else 50

        # Set up generator for reproducibility
        if seed == -1:
            generator = None  # Let torch pick a random seed
        else:
            generator = torch.Generator(device=self._resolved_device).manual_seed(seed)

        # Ensure inputs are correct mode and size
        image_rgb = image.convert("RGB")
        mask_gray = mask.convert("L")
        if mask_gray.size != image_rgb.size:
            mask_gray = mask_gray.resize(image_rgb.size, Image.LANCZOS)

        logger.debug(
            "Inpainting: size=%s denoise=%.2f steps=%d seed=%s device=%s",
            image_rgb.size,
            denoise,
            num_steps,
            seed,
            self._resolved_device,
        )

        try:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_rgb,
                mask_image=mask_gray,
                strength=denoise,
                num_inference_steps=num_steps,
                generator=generator,
            ).images[0]
        except Exception as exc:
            raise RuntimeError(
                f"SD1.5 inpainting failed on {self._resolved_device}: {exc}"
            ) from exc

        return result
