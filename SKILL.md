# photo-edit — AI Photo Editing Skill

**When to use:** Any request involving removing people/objects from photos, inpainting regions, cleaning up backgrounds, or editing photo content with AI.

Trigger phrases: "remove the person", "edit this photo", "clean up the background", "take out X from the photo", "inpaint", "fill in the background".

---

## What This Skill Does

- **Object/person removal** from photos using Stable Diffusion inpainting
- **Face protection** — detects all faces, protects bystanders from inpainting damage
- **Multi-variant generation** — runs N times, ranks by quality, returns best
- **Quality scoring** — SSIM-based face preservation check with pass/fail

---

## CLI Syntax

```bash
cd /Users/aven/.openclaw/workspace/skills/photo-edit

# Basic usage — provide a mask image
python pipeline.py \
  --input photo.jpg \
  --mask mask.png \
  --output out/ \
  --variants 3 \
  --denoise 0.25

# With a fill prompt (what to put in the removed area)
python pipeline.py \
  --input group.jpg \
  --mask person2_mask.png \
  --prompt "park bench, natural background" \
  --output out/ \
  --variants 5

# Aggressive removal (more creative fill, less context preservation)
python pipeline.py \
  --input photo.jpg \
  --mask mask.png \
  --denoise 0.75 \
  --output out/

# Use CPU if MPS has issues
python pipeline.py --input photo.jpg --mask mask.png --output out/ --device cpu

# Verbose logging for debugging
python pipeline.py --input photo.jpg --mask mask.png --output out/ --verbose
```

### All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Source photo path |
| `--output` | required | Output directory |
| `--mask` | required* | Mask image path (white=remove, black=keep) |
| `--mask-mode auto` | — | Auto-generate mask (not yet implemented) |
| `--prompt` | `""` | What to fill the masked area with |
| `--negative-prompt` | backend default | What to avoid generating |
| `--variants` | `3` | How many variants to generate |
| `--denoise` | `0.25` | Inpainting strength 0.0–1.0 |
| `--backend` | `sd15_inpaint` | Backend to use |
| `--device` | `mps` | Compute device: mps / cuda / cpu |
| `--face-ssim-threshold` | `0.75` | Face preservation quality threshold |
| `--face-padding` | `20` | Padding around protected face boxes |
| `--verbose` | off | Enable DEBUG logging |

*Either `--mask` or `--mask-mode` is required.

---

## Creating Masks

A mask is a **grayscale image the same size as the photo**:
- **White (255)** = region to inpaint (remove/replace)
- **Black (0)** = region to keep exactly

### Option A: Create a mask with Preview (macOS)

1. Open photo in Preview
2. Markup toolbar → Instant Alpha or selection tool
3. Select the person/object to remove
4. Export selection as PNG with transparent background
5. Convert to grayscale mask: `python3 -c "from PIL import Image; img=Image.open('sel.png'); mask=img.split()[3]; mask.save('mask.png')"`

### Option B: Create a mask in code

```python
from processors.masking import create_ellipse_mask, dilate_mask

# Quick circular mask around a person
mask = create_ellipse_mask(
    image_size=(1920, 1080),
    center=(640, 540),    # pixel center of the person
    radius=(150, 300),    # (width_radius, height_radius) for ellipse
)
mask = dilate_mask(mask, pixels=15)  # Feather the edges
mask.save("mask.png")
```

### Option C: Draw in Photoshop / GIMP

Paint white over the region to remove, save as grayscale PNG.

---

## Interpreting JSON Output

The pipeline prints JSON to stdout:

```json
{
  "best": "/path/to/out/photo_variant_01_seed12345.png",
  "variants": [
    "/path/to/out/photo_variant_01_seed12345.png",
    "/path/to/out/photo_variant_02_seed67890.png",
    "/path/to/out/photo_variant_03_seed11111.png"
  ],
  "scores": [
    {"face_ssim": 0.923, "passed": true, "warnings": [], "score": 0.923},
    {"face_ssim": 0.891, "passed": true, "warnings": [], "score": 0.891},
    {"face_ssim": 0.612, "passed": false, "warnings": ["face_distortion (ssim=0.612 < threshold=0.75)"], "score": 0.612}
  ],
  "warnings": [],
  "metadata": {
    "input": "group.jpg",
    "total_faces": 4,
    "protected_faces": 3,
    "variants_requested": 3,
    "variants_generated": 3,
    "denoise": 0.25,
    "backend": "runwayml/stable-diffusion-inpainting"
  }
}
```

**Key fields:**
- `best` — Path to the highest-quality variant (show this to the user)
- `scores[N].passed` — Whether variant N passed quality checks
- `scores[N].face_ssim` — 0.0–1.0 face preservation score (>0.75 = good)
- `warnings` — Global warnings (non-empty = no variant passed quality bar)
- `metadata.protected_faces` — How many faces were shielded from inpainting

**Exit codes:**
- `0` — At least one variant passed quality checks
- `1` — No variants passed (best returned with WARNING)
- `2` — Fatal error (bad input, missing model, etc.)

---

## First Run — Model Download

On first run, the SD1.5 inpainting model (~4 GB) downloads automatically.
Pre-download with:

```bash
python3 -c "
from diffusers import StableDiffusionInpaintPipeline
StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting')
print('Model cached!')
"
```

---

## Tuning Denoise Strength

| Denoise | Use When |
|---------|----------|
| `0.15–0.30` | Subject is close to others, need subtle fill |
| `0.40–0.60` | Standard removal, medium complexity background |
| `0.70–0.85` | Complex fill needed (faces, detailed backgrounds) |
| `0.90–1.00` | Full creative freedom (loses original context) |

Start at `0.25`. If the removed region looks too "stamped in", increase to `0.5`.

---

## Swapping Backends (Upgrade Path)

The backend is abstracted behind `backends/base.py`. To add a new backend:

### 1. Create `backends/flux_fill.py`

```python
from .base import InpaintingBackend

class FluxFillBackend(InpaintingBackend):
    model_id = "black-forest-labs/FLUX.1-Fill-dev"

    def inpaint(self, image, mask, prompt, negative_prompt, denoise, seed):
        # FLUX Fill implementation
        ...

    def is_available(self):
        ...
```

### 2. Register in `backends/__init__.py`

```python
from .flux_fill import FluxFillBackend

BACKEND_REGISTRY["flux_fill"] = FluxFillBackend
```

### 3. Use it

```bash
python pipeline.py --input photo.jpg --mask mask.png --backend flux_fill --output out/
```

### Backend Comparison

| Backend | Quality | Speed | VRAM | Status |
|---------|---------|-------|------|--------|
| `sd15_inpaint` | Good | Fast (MPS) | ~4 GB | ✅ Implemented |
| `flux_fill` | Excellent | Medium | ~12 GB | TODO |
| `pulid` | Excellent (face-consistent) | Slow | ~16 GB | TODO |
| `cloud_replicate` | Varies | Fast | 0 (API) | TODO |

---

## Programmatic Use (Python)

```python
from pathlib import Path
from pipeline import Pipeline

pipeline = Pipeline(
    backend_name="sd15_inpaint",
    device="mps",
    face_ssim_threshold=0.75,
)

summary = pipeline.run(
    input_path=Path("group.jpg"),
    output_dir=Path("out/"),
    mask_path=Path("mask.png"),
    prompt="park bench, summer day",
    variants=3,
    denoise=0.30,
)

print(f"Best result: {summary['best']}")
print(f"Quality scores: {summary['scores']}")
```

---

## Troubleshooting

**"MPS backend out of memory"**
→ Use `--device cpu` (slower) or reduce image resolution before inpainting.

**Face SSIM always low (<0.6)**
→ Try `--denoise 0.15` — more conservative fill.
→ Try `--face-ssim-threshold 0.65` to relax the threshold.

**Visible seam at mask edge**
→ Dilate your mask by 10–15px to include more context.
→ Try a slightly higher denoise to let the model blend more.

**Model download fails**
→ Ensure Hugging Face Hub is reachable, or set `HF_HUB_OFFLINE=1` after caching.

**`mediapipe` install fails on ARM**
→ `pip install mediapipe --extra-index-url https://google-coral.github.io/py-repo/`
→ Or: `conda install -c conda-forge mediapipe`
