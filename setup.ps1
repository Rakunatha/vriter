$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "  Vrite - Windows Setup"
Write-Host "========================================"
Write-Host ""

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[FAIL] Python not found. Download from python.org"
    exit 1
}
Write-Host "[OK]   Python found: $(python --version)"

$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) {
    Write-Host "[FAIL] ffmpeg not found. Download from ffmpeg.org"
    Write-Host "       Add the bin\ folder to your system PATH."
    exit 1
}
Write-Host "[OK]   ffmpeg found"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "[OK]   Virtual environment created"
}

. .\.venv\Scripts\Activate.ps1
Write-Host "[OK]   Virtual environment active"

$hasGpu = $false
try {
    $null = nvidia-smi 2>$null
    $hasGpu = $true
} catch {}

if ($hasGpu) {
    Write-Host "[OK]   NVIDIA GPU detected - installing CUDA build"
    pip install --no-cache-dir torch torchvision torchaudio `
        --index-url https://download.pytorch.org/whl/cu118 --quiet
} else {
    Write-Host "[WARN] No GPU - installing CPU build"
    pip install --no-cache-dir torch torchvision torchaudio `
        --index-url https://download.pytorch.org/whl/cpu --quiet
}

Write-Host "[OK]   Installing project dependencies ..."
pip install --no-cache-dir -r requirements.txt --quiet
Write-Host "[OK]   Dependencies installed"

Write-Host "[OK]   Downloading model weights ..."
python -m vrite.pipeline.model_downloader

New-Item -ItemType Directory -Force -Path "uploads"  | Out-Null
New-Item -ItemType Directory -Force -Path "outputs"  | Out-Null

Write-Host ""
Write-Host "========================================"
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  Activate venv (each session):"
Write-Host "    . .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Web UI:  streamlit run ui\app.py"
Write-Host "  CLI:     python run.py --help"
Write-Host "========================================"
Write-Host ""
