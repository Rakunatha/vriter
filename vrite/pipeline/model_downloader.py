"""
Vrite - Model Downloader
One-time setup: clones repos and downloads model weights.

Usage:
    python -m vrite.pipeline.model_downloader
"""
from __future__ import annotations

import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger("vrite.downloader")

MODELS_DIR    = Path("models")
WAV2LIP_DIR   = Path("Wav2Lip")
SADTALKER_DIR = Path("SadTalker")

CHECKPOINTS = {
    "wav2lip_gan": {
        "dest": MODELS_DIR / "wav2lip_gan.pth",
        "urls": [
            "https://huggingface.co/numz/wav2lip_studio/resolve/"
            "main/Wav2lip/wav2lip_gan.pth",
        ],
        "size_mb": 430,
        "note": "Wav2Lip GAN checkpoint",
    },
    "s3fd": {
        "dest": (WAV2LIP_DIR / "face_detection"
                 / "detection" / "sfd" / "s3fd.pth"),
        "urls": [
            "https://www.adrianbulat.com/downloads/"
            "python-fan/s3fd-619a316812.pth",
        ],
        "size_mb": 85,
        "note": "S3FD face detector",
    },
}

SADTALKER_FILES = {
    "SadTalker_V0.0.2_256.safetensors": (
        "https://github.com/OpenTalker/SadTalker/releases/download/"
        "v0.0.2-rc/SadTalker_V0.0.2_256.safetensors"),
    "mapping_00109-model.pth.tar": (
        "https://github.com/OpenTalker/SadTalker/releases/download/"
        "v0.0.2-rc/mapping_00109-model.pth.tar"),
    "mapping_00229-model.pth.tar": (
        "https://github.com/OpenTalker/SadTalker/releases/download/"
        "v0.0.2-rc/mapping_00229-model.pth.tar"),
}


class _Progress:
    def __call__(self, block, block_size, total):
        pct = int(block * block_size * 100 / max(total, 1))
        sys.stdout.write(f"\r    {min(pct,100):3d}%")
        sys.stdout.flush()


def _clone(url: str, dest: Path, name: str) -> None:
    if dest.exists():
        print(f"  [OK] {name} already cloned")
        return
    print(f"  Cloning {name} ...")
    subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)], check=True)
    print(f"  [OK] {name}")


def _download(url: str, dest: Path, label: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [OK] {label or dest.name} already exists")
        return True
    print(f"  Downloading {label or dest.name} ...")
    try:
        urllib.request.urlretrieve(url, str(dest), _Progress())
        print()
        print(f"  [OK] {dest}")
        return True
    except Exception as exc:
        print(f"\n  [FAIL] {exc}")
        if dest.exists():
            dest.unlink()
        return False


def run_setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "sadtalker").mkdir(exist_ok=True)

    print("\n" + "="*48)
    print("  Vrite - First-time Model Setup")
    print("="*48)

    print("\n[1/4] Cloning repositories ...")
    _clone("https://github.com/Rudrabha/Wav2Lip.git",
           WAV2LIP_DIR, "Wav2Lip")
    _clone("https://github.com/OpenTalker/SadTalker.git",
           SADTALKER_DIR, "SadTalker")

    print("\n[2/4] Downloading Wav2Lip checkpoints ...")
    for key, info in CHECKPOINTS.items():
        print(f"\n  [{info['note']}  ~{info['size_mb']} MB]")
        ok = any(_download(url, info["dest"], info["note"])
                 for url in info["urls"])
        if not ok:
            print(f"  [WARN] Could not auto-download {key}.")

    print("\n[3/4] Downloading SadTalker checkpoints ...")
    for fname, url in SADTALKER_FILES.items():
        _download(url, MODELS_DIR / "sadtalker" / fname, fname)

    print("\n[4/4] Installing Wav2Lip requirements ...")
    req = WAV2LIP_DIR / "requirements.txt"
    if req.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "-r", str(req), "-q"], check=True)

    print("\n" + "="*48)
    print("  [OK] Setup complete!")
    print("="*48)
    print("\n  Web UI:  streamlit run ui/app.py")
    print("  CLI:     python run.py --help\n")


if __name__ == "__main__":
    run_setup()
