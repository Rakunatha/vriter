#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "========================================"
echo "  Vrite - Setup Script"
echo "========================================"
echo ""

PYTHON=$(command -v python3.11 \
      || command -v python3 \
      || command -v python \
      || true)
[ -z "$PYTHON" ] && echo "[FAIL] Python not found" && exit 1
echo "[OK]   Python: $($PYTHON --version)"

command -v ffmpeg >/dev/null 2>&1 \
  || { echo "[FAIL] ffmpeg not found - install: sudo apt install ffmpeg"; exit 1; }
echo "[OK]   ffmpeg found"

if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo "[OK]   Virtual environment created"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[OK]   Virtual environment active"

GPU=false
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -q "GPU" && GPU=true
fi

if $GPU; then
    echo "[OK]   NVIDIA GPU detected - installing CUDA build"
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "[WARN] No GPU - installing CPU build (slower generation)"
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

echo "[OK]   Installing project dependencies ..."
pip install --no-cache-dir -r requirements.txt -q
echo "[OK]   Dependencies installed"

echo "[OK]   Downloading model weights ..."
python -m vrite.pipeline.model_downloader

mkdir -p uploads outputs

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Web UI:  streamlit run ui/app.py"
echo "  CLI:     python run.py --help"
echo "========================================"
echo ""
