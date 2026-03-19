"""
Vrite - Video Enhancer (optional)
Upscaling, face restore, sharpening, frame interpolation, LUT grading.
Each step skips gracefully if its model is not installed.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vrite.enhancer")

CODEFORMER_DIR = Path("CodeFormer")
REALESRGAN_DIR = Path("Real-ESRGAN")


class VideoEnhancer:
    def __init__(self, upscale: int = 1,
                 face_restore: bool = False,
                 interpolate: bool = False,
                 sharpen: bool = True,
                 lut_path: str = None,
                 device: str = "cpu"):
        self.upscale = upscale
        self.face_restore = face_restore
        self.interpolate = interpolate
        self.sharpen = sharpen
        self.lut_path = lut_path
        self.device = device

    def enhance(self, input_path: str, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cur = input_path

        if (self.face_restore
                and (CODEFORMER_DIR / "inference_codeformer.py").exists()):
            cur = self._codeformer(cur, output_path + "_cf.mp4")
        if self.upscale > 1 and self._realesrgan_available():
            cur = self._realesrgan(cur, output_path + "_sr.mp4")
        if self.sharpen:
            cur = self._sharpen(cur, output_path + "_sh.mp4")
        if self.interpolate:
            cur = self._interpolate(cur, output_path + "_in.mp4")
        if self.lut_path and Path(self.lut_path).exists():
            cur = self._lut(cur, self.lut_path, output_path + "_lt.mp4")

        if cur != output_path:
            shutil.move(cur, output_path)
        for suf in ["_cf.mp4", "_sr.mp4", "_sh.mp4", "_in.mp4", "_lt.mp4"]:
            p = Path(output_path + suf)
            if p.exists():
                p.unlink()

        log.info("Enhancement complete -> %s", output_path)
        return output_path

    def _codeformer(self, video: str, out: str) -> str:
        r = subprocess.run(
            [sys.executable,
             str(CODEFORMER_DIR / "inference_codeformer.py"),
             "--input_path", video,
             "--output_path", str(Path(out).parent),
             "--face_upsample", "--fidelity_weight", "0.7"],
            text=True)
        if r.returncode != 0:
            log.warning("CodeFormer failed - skipping")
            return video
        results = sorted(Path(out).parent.glob("*.mp4"))
        return str(results[-1]) if results else video

    def _realesrgan(self, video: str, out: str) -> str:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            log.warning("realesrgan not installed - skipping")
            return video
        name = "RealESRGAN_x4plus" if self.upscale >= 4 else "RealESRGAN_x2plus"
        mp = f"models/{name}.pth"
        if not Path(mp).exists():
            log.warning("Real-ESRGAN model not found - skipping")
            return video
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=self.upscale)
        up = RealESRGANer(scale=self.upscale, model_path=mp,
                          model=model, half=False, device=self.device)
        return self._frame_fn(video, out, up.enhance)

    @staticmethod
    def _realesrgan_available() -> bool:
        try:
            import realesrgan
            return True
        except ImportError:
            return False

    @staticmethod
    def _sharpen(video: str, out: str) -> str:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video,
             "-vf",
             "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8",
             "-c:a", "copy", out],
            check=True, capture_output=True)
        return out

    @staticmethod
    def _interpolate(video: str, out: str) -> str:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video,
             "-vf",
             "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
             "-c:a", "copy", out],
            check=True, capture_output=True)
        return out

    @staticmethod
    def _lut(video: str, lut: str, out: str) -> str:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video,
             "-vf", f"lut3d='{lut}'",
             "-c:a", "copy", out],
            check=True, capture_output=True)
        return out

    @staticmethod
    def _frame_fn(video: str, out: str, fn) -> str:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = out.replace(".mp4", "_silent.mp4")
        wr = cv2.VideoWriter(
            tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                enhanced, _ = fn(frame[:, :, ::-1])
                enhanced = enhanced[:, :, ::-1]
                if enhanced.shape[:2] != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                wr.write(enhanced)
            except Exception:
                wr.write(frame)
        cap.release()
        wr.release()
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", tmp, "-i", video,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-crf", "18",
             "-c:a", "copy", out],
            check=True, capture_output=True)
        Path(tmp).unlink(missing_ok=True)
        return out
