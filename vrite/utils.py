"""
Vrite - Shared utilities
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S")
    if not debug:
        for noisy in ("urllib3", "requests", "numba", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs:02d}s" if mins else f"{secs}s"


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    if not check_ffmpeg():
        raise EnvironmentError(
            "ffmpeg not found. Install: sudo apt install ffmpeg")


def safe_tmp_path(tmp_dir: str, name: str) -> str:
    return str(Path(tmp_dir).resolve() / name)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ensure_mono_wav(input_path: str, output_path: str,
                    sample_rate: int = 16000) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ac", "1", "-ar", str(sample_rate),
         "-acodec", "pcm_s16le", output_path],
        check=True, capture_output=True)
    return output_path


def trim_audio(input_path: str, output_path: str,
               max_seconds: float) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-t", str(max_seconds), "-c", "copy", output_path],
        check=True, capture_output=True)
    return output_path
