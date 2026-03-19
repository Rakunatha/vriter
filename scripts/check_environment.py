"""
Vrite - Environment Checker
Validates all dependencies before first run.

Usage:
    python scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK   = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"


def ok(label: str, detail: str = "") -> None:
    print(f"  {OK} {label:<40s} {detail}")


def warn(label: str, detail: str = "") -> None:
    print(f"  {WARN} {label:<40s} {detail}")


def fail(label: str, detail: str = "") -> None:
    print(f"  {FAIL} {label:<40s} {detail}")


def section(title: str) -> None:
    print(f"\n  {'-'*55}")
    print(f"  {title}")
    print(f"  {'-'*55}")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        ok("Python version", ver)
    else:
        fail("Python version", f"{ver} - need 3.10+")


def check_system_tools() -> None:
    section("System tools")
    for tool, note in [
        ("ffmpeg",  "Required for all video I/O"),
        ("ffprobe", "Required for metadata probing"),
        ("git",     "Required for model downloads"),
    ]:
        if shutil.which(tool):
            r = subprocess.run(
                [tool, "-version"], capture_output=True, text=True)
            ver = (r.stdout.splitlines()[0][:55]
                   if r.stdout else "?")
            ok(tool, ver)
        else:
            fail(tool, note)


def check_packages() -> None:
    section("Python packages (required)")
    required = [
        ("torch",      "Deep learning runtime"),
        ("cv2",        "Computer vision"),
        ("numpy",      "Numerical computing"),
        ("scipy",      "Signal processing"),
        ("soundfile",  "Audio I/O"),
        ("librosa",    "Audio analysis"),
        ("ffmpeg",     "ffmpeg-python bindings"),
        ("streamlit",  "Web UI"),
    ]
    for pkg, note in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(pkg, f"v{ver}")
        except ImportError:
            fail(pkg, note)

    section("Python packages (optional)")
    optional = [
        ("TTS",        "Coqui XTTS-v2 voice cloning"),
        ("gtts",       "gTTS cloud TTS (free tier)"),
        ("pyttsx3",    "Offline system TTS fallback"),
        ("safetensors","SadTalker checkpoint format"),
    ]
    for pkg, note in optional:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg}", f"v{ver}  ({note})")
        except ImportError:
            warn(f"{pkg}", note)


def check_gpu() -> None:
    section("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = (torch.cuda.get_device_properties(0)
                   .total_memory // (1024**3))
            ok("CUDA available", f"{name}  ({mem} GB VRAM)")
        else:
            warn("CUDA not available",
                 "CPU mode - generation will be slower")
    except ImportError:
        fail("PyTorch not installed", "pip install torch")


def check_repos() -> None:
    section("External repositories")
    for name, check_file, cmd in [
        ("Wav2Lip",
         "Wav2Lip/inference.py",
         "git clone https://github.com/Rudrabha/Wav2Lip.git"),
        ("SadTalker",
         "SadTalker/inference.py",
         "git clone https://github.com/OpenTalker/SadTalker.git"),
    ]:
        if (ROOT / check_file).exists():
            ok(name, "Cloned")
        else:
            warn(name, cmd)


def check_models() -> None:
    section("Model weights")
    models = [
        ("models/wav2lip_gan.pth",
         "Wav2Lip GAN checkpoint (~430 MB)"),
        ("Wav2Lip/face_detection/detection/sfd/s3fd.pth",
         "S3FD face detector (~85 MB)"),
        ("models/sadtalker/SadTalker_V0.0.2_256.safetensors",
         "SadTalker weights (~300 MB)"),
    ]
    for path, note in models:
        p = ROOT / path
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            ok(p.name, f"{mb:.0f} MB")
        else:
            warn(p.name,
                 f"Missing - run: python -m vrite.pipeline.model_downloader")


def main() -> None:
    print("\n" + "="*58)
    print("  Vrite - Environment Check")
    print("="*58)

    check_python()
    check_system_tools()
    check_packages()
    check_gpu()
    check_repos()
    check_models()

    print("\n" + "="*58)
    print("  Fix any [FAIL] items before running.")
    print("  [WARN] items are optional enhancements.")
    print("="*58 + "\n")


if __name__ == "__main__":
    main()
