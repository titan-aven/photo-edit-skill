#!/usr/bin/env bash
# setup.sh — Install dependencies and print device/readiness info
#
# Usage:
#   bash setup.sh
#
# Installs all Python requirements and checks whether your hardware is
# ready for local inference. Does NOT download any model weights.
# Model weights are downloaded automatically on first pipeline.py run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Photo Edit Skill — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# ── Python version check ──────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Python: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    echo "⚠️  Python 3.10+ recommended. You have $PYTHON_VERSION."
    echo "   Things may still work, but type hints and some features require 3.10+."
fi

# ── Install dependencies ──────────────────────────────────────────────────
echo
echo "Installing requirements…"
pip3 install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Device Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - <<'PYEOF'
import sys

# ── Torch + device availability ────────────────────────────────────────
try:
    import torch
    print(f"torch: {torch.__version__}")

    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    if mps_ok:
        print("✅ MPS (Apple Silicon): available — will use GPU acceleration")
        print("   Recommended --device mps")
    elif cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA: available ({gpu_name})")
        print("   Recommended --device cuda")
    else:
        print("⚠️  No GPU detected — will use CPU (slow, ~5-15 min per variant)")
        print("   Recommended --device cpu")

except ImportError:
    print("❌ torch not installed — run `pip install torch` or check requirements.txt")
    sys.exit(1)

# ── diffusers ─────────────────────────────────────────────────────────
try:
    import diffusers
    print(f"\ndiffusers: {diffusers.__version__}")
except ImportError:
    print("\n❌ diffusers not installed")

# ── mediapipe ─────────────────────────────────────────────────────────
try:
    import mediapipe
    print(f"mediapipe: {mediapipe.__version__}")
except ImportError:
    print("❌ mediapipe not installed")

# ── Pillow ────────────────────────────────────────────────────────────
try:
    from PIL import Image
    import PIL
    print(f"Pillow: {PIL.__version__}")
except ImportError:
    print("❌ Pillow not installed")

# ── scikit-image ──────────────────────────────────────────────────────
try:
    import skimage
    print(f"scikit-image: {skimage.__version__}")
except ImportError:
    print("❌ scikit-image not installed")

PYEOF

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model Download Instructions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Model: runwayml/stable-diffusion-inpainting"
echo "Size:  ~4 GB (cached at ~/.cache/huggingface/hub/)"
echo
echo "The model downloads automatically on first pipeline.py run."
echo "To pre-download manually:"
echo
echo "  python3 - <<'EOF'"
echo "  from diffusers import StableDiffusionInpaintPipeline"
echo "  StableDiffusionInpaintPipeline.from_pretrained("
echo "      'runwayml/stable-diffusion-inpainting'"
echo "  )"
echo "  print('Model cached!')"
echo "  EOF"
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Quick Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "To verify the pipeline loads correctly (no model download):"
echo
echo "  cd $SCRIPT_DIR"
echo "  python3 -c \"from pipeline import Pipeline; print('Pipeline imports OK')\""
echo
echo "Setup complete. Run 'python3 pipeline.py --help' to get started."
