"""
Photo editing pipeline — CLI entry point and orchestrator.

Usage:
    python pipeline.py --input photo.jpg --mask mask.png --output out/ --variants 3
    python pipeline.py --input photo.jpg --mask-mode auto --prompt "fill background" --output out/

The pipeline wires together the backend and processors:
  1. Load image + detect ALL faces
  2. Load or generate the removal mask
  3. Create composite mask: removal zone ∪ (removal zone − protected faces)
  4. Run the backend N times (each with a different random seed)
  5. Quality-check each output
  6. Return ranked variants as JSON to stdout

Exit codes:
  0 — At least one variant passed quality checks
  1 — No variants passed (best is returned with WARNING in JSON)
  2 — Fatal error (bad input, missing model, etc.)

Design note:
  The Pipeline class is instantiatable for programmatic use.
  The CLI at the bottom is just a thin argparse wrapper around Pipeline.run().
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

# Configure logging — INFO goes to stderr so it doesn't pollute JSON stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end photo editing orchestrator.

    Separates orchestration logic from backend and processor implementations
    so that each piece can be tested and swapped independently.
    """

    def __init__(
        self,
        backend_name: str = "sd15_inpaint",
        device: str = "mps",
        face_ssim_threshold: float = 0.75,
    ) -> None:
        """
        Args:
            backend_name:        Key in backends.BACKEND_REGISTRY.
            device:              Compute device for the backend.
            face_ssim_threshold: Pass/fail threshold for face preservation.
        """
        from backends import BACKEND_REGISTRY
        from processors.quality_check import QualityChecker

        if backend_name not in BACKEND_REGISTRY:
            available = ", ".join(BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend: {backend_name!r}. Available: {available}"
            )

        self.backend = BACKEND_REGISTRY[backend_name](device=device)
        self.checker = QualityChecker(face_ssim_threshold=face_ssim_threshold)
        logger.info("Pipeline ready: backend=%s device=%s", backend_name, device)

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        mask_path: Path | None = None,
        mask_mode: str | None = None,
        prompt: str = "",
        negative_prompt: str = "",
        variants: int = 3,
        denoise: float = 0.25,
        face_padding: int = 20,
    ) -> dict[str, Any]:
        """
        Execute the full pipeline and return a JSON-serializable summary.

        Args:
            input_path:       Path to the source photo.
            output_dir:       Directory where variants are saved.
            mask_path:        Path to a pre-made mask. Used if mask_mode is None.
            mask_mode:        "auto" = generate mask from prompt (stub; see TODO).
            prompt:           What to fill the inpainted region with.
            negative_prompt:  What to avoid. Uses backend default if empty.
            variants:         Number of variants to generate.
            denoise:          Inpainting strength [0.0, 1.0].
            face_padding:     Padding pixels around protected face boxes.

        Returns:
            Dict with keys: best, variants, scores, warnings.
            The dict is printed as JSON to stdout by the CLI.

        Raises:
            FileNotFoundError: If input_path or mask_path doesn't exist.
            ValueError:        If arguments are invalid.
            RuntimeError:      If the backend fails.
        """
        from processors.face_guard import FaceGuard
        from processors.masking import load_mask

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Load source image ──────────────────────────────────────
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        logger.info("Loading image: %s", input_path)
        image = Image.open(input_path).convert("RGB")
        logger.info("Image size: %dx%d", *image.size)

        # ── Step 2: Detect all faces in the source image ──────────────────
        guard = FaceGuard()
        logger.info("Detecting faces…")
        all_faces = guard.detect_faces(image)
        logger.info("Found %d face(s)", len(all_faces))

        # ── Step 3: Load or create the removal mask ───────────────────────
        if mask_path is not None:
            logger.info("Loading mask: %s", mask_path)
            removal_mask = load_mask(mask_path, target_size=image.size)
        elif mask_mode == "auto":
            # TODO: Implement auto-mask via SAM (Segment Anything) or
            # a text-grounded segmentation model. For now, raise to avoid
            # silently doing nothing useful.
            raise NotImplementedError(
                "--mask-mode auto is not yet implemented. "
                "Provide a mask with --mask, or implement SAM integration."
            )
        else:
            raise ValueError("Either --mask or --mask-mode must be provided.")

        # ── Step 4: Create composite mask (protect innocent bystanders) ───
        logger.info("Creating face-protected mask…")
        protected_mask = guard.create_face_protection_mask(
            image, removal_mask, padding=face_padding
        )

        # Identify which faces are protected (outside the removal zone)
        protected_faces = [f for f in all_faces if not f.overlaps_mask(removal_mask)]
        logger.info("%d face(s) protected from inpainting", len(protected_faces))

        # ── Step 5: Run backend N times ────────────────────────────────────
        logger.info(
            "Generating %d variant(s) with denoise=%.2f…", variants, denoise
        )

        neg_prompt = negative_prompt or ""  # Backend uses its own default if empty

        generated: list[tuple[Image.Image, int]] = []
        for i in range(variants):
            seed = random.randint(0, 2**31 - 1)
            logger.info("Variant %d/%d (seed=%d)…", i + 1, variants, seed)
            try:
                result = self.backend.inpaint(
                    image=image,
                    mask=protected_mask,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    denoise=denoise,
                    seed=seed,
                )
                generated.append((result, seed))
            except Exception as exc:
                logger.error("Variant %d failed: %s", i + 1, exc)
                # Continue trying remaining variants instead of aborting

        if not generated:
            raise RuntimeError("All variants failed to generate. See stderr for details.")

        # ── Step 6: Quality-check each variant ─────────────────────────────
        logger.info("Running quality checks…")
        scored: list[tuple[Image.Image, int, object]] = []
        for img, seed in generated:
            qs = self.checker.score(image, img, protected_faces)
            scored.append((img, seed, qs))

        # ── Step 7: Rank variants, save to disk ───────────────────────────
        # rank_variants expects (image, score) pairs
        ranked = self.checker.rank_variants([(img, qs) for img, _, qs in scored])

        stem = input_path.stem
        saved_paths: list[str] = []
        score_dicts: list[dict] = []

        for rank, (img, qs) in enumerate(ranked):
            # Find seed for this image (by identity)
            seed = next(s for i, s, _ in scored if i is img)
            out_path = output_dir / f"{stem}_variant_{rank + 1:02d}_seed{seed}.png"
            img.save(out_path, format="PNG")
            saved_paths.append(str(out_path))
            score_dicts.append(qs.to_dict())
            logger.info(
                "Saved variant %d → %s (%s)",
                rank + 1,
                out_path.name,
                "PASS" if qs.passed else "FAIL",
            )

        # ── Assemble JSON summary ──────────────────────────────────────────
        global_warnings: list[str] = []
        best_passed = score_dicts[0]["passed"] if score_dicts else False

        if not best_passed:
            global_warnings.append(
                "WARNING: No variants passed quality thresholds. "
                "Best result returned but may have face distortion."
            )
            logger.warning("No passing variants — returning best-effort result")

        summary: dict[str, Any] = {
            "best": saved_paths[0] if saved_paths else None,
            "variants": saved_paths,
            "scores": score_dicts,
            "warnings": global_warnings,
            "metadata": {
                "input": str(input_path),
                "total_faces": len(all_faces),
                "protected_faces": len(protected_faces),
                "variants_requested": variants,
                "variants_generated": len(generated),
                "denoise": denoise,
                "backend": self.backend.model_id,
            },
        }

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser():
    """Build and return the argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI photo inpainting pipeline — remove objects/people from photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove a person using a pre-made mask
  python pipeline.py --input group.jpg --mask person_mask.png --output out/ --variants 3

  # Aggressive removal with more creative fill
  python pipeline.py --input photo.jpg --mask mask.png --denoise 0.7 --output out/

  # Use a specific backend
  python pipeline.py --input photo.jpg --mask mask.png --backend sd15_inpaint --output out/

Output: JSON to stdout with best variant path and quality scores.
        """,
    )

    parser.add_argument("--input", required=True, help="Source photo path")
    parser.add_argument("--output", required=True, help="Output directory")

    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("--mask", help="Path to inpainting mask (white=remove)")
    mask_group.add_argument(
        "--mask-mode",
        choices=["auto"],
        help="Auto-generate mask. 'auto' = derive from --prompt",
    )

    parser.add_argument("--prompt", default="", help="Fill prompt (what to put in masked area)")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt (what to avoid). Uses backend default if unset.",
    )
    parser.add_argument("--variants", type=int, default=3, help="Number of variants to generate (default: 3)")
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.25,
        help="Inpainting strength 0.0-1.0 (default: 0.25). "
             "Low=conservative, High=creative.",
    )
    parser.add_argument(
        "--backend",
        default="sd15_inpaint",
        help="Backend name. One of: sd15_inpaint (default)",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Compute device (default: mps for Apple Silicon)",
    )
    parser.add_argument(
        "--face-ssim-threshold",
        type=float,
        default=0.75,
        help="Minimum face SSIM to pass quality check (default: 0.75)",
    )
    parser.add_argument(
        "--face-padding",
        type=int,
        default=20,
        help="Padding pixels around protected face boxes (default: 20)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")

    return parser


def main() -> int:
    """
    CLI entry point.

    Returns exit code: 0 = success, 1 = warning (no pass), 2 = fatal error.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        pipeline = Pipeline(
            backend_name=args.backend,
            device=args.device,
            face_ssim_threshold=args.face_ssim_threshold,
        )

        summary = pipeline.run(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            mask_path=Path(args.mask) if args.mask else None,
            mask_mode=args.mask_mode,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            variants=args.variants,
            denoise=args.denoise,
            face_padding=args.face_padding,
        )

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        print(json.dumps({"error": str(exc), "code": "FILE_NOT_FOUND"}))
        return 2
    except ValueError as exc:
        logger.error("Invalid argument: %s", exc)
        print(json.dumps({"error": str(exc), "code": "INVALID_ARGUMENT"}))
        return 2
    except NotImplementedError as exc:
        logger.error("Not implemented: %s", exc)
        print(json.dumps({"error": str(exc), "code": "NOT_IMPLEMENTED"}))
        return 2
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        print(json.dumps({"error": str(exc), "code": "RUNTIME_ERROR"}))
        return 2

    # Print JSON summary to stdout (clean, no logging noise)
    print(json.dumps(summary, indent=2))

    # Exit 1 if no variants passed quality checks
    all_failed = summary.get("warnings") and any(
        "WARNING" in w for w in summary["warnings"]
    )
    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
