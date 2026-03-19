"""
Vrite - Lip-Sync Engine
Priority: Wav2Lip GAN -> SadTalker -> audio-swap fallback
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vrite.config import PipelineConfig
from vrite.utils import safe_tmp_path, ensure_mono_wav

log = logging.getLogger("vrite.lipsync")

WAV2LIP_DIR   = Path("Wav2Lip")
SADTALKER_DIR = Path("SadTalker")


class LipSyncEngine:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def process(self, source_video: str, driven_audio: str,
                style_meta: dict[str, Any],
                out_dir: str = None) -> str:
        out_dir_p = Path(out_dir or self.cfg.tmp_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_p / "lipsync_out.mp4")

        norm = safe_tmp_path(str(out_dir_p), "driven_norm.wav")
        ensure_mono_wav(driven_audio, norm)

        if (Path(self.cfg.wav2lip_checkpoint).exists()
                and WAV2LIP_DIR.exists()):
            log.info("Strategy: Wav2Lip GAN")
            return self._wav2lip(source_video, norm, out_path)

       if (SADTALKER_DIR / "inference.py").exists() and self._sadtalker_ready():
    log.info("Strategy: SadTalker")
    return self._sadtalker(source_video, norm, out_path)

        log.warning("No lip-sync model - using audio-swap fallback")
        return self._audio_swap(source_video, norm, out_path)

    def _wav2lip(self, video: str, audio: str,
                 out_path: str) -> str:
        top, bot, left, right = self.cfg.wav2lip_pads
        cmd = [
            sys.executable,
            str(WAV2LIP_DIR / "inference.py"),
            "--checkpoint_path",     self.cfg.wav2lip_checkpoint,
            "--face",                video,
            "--audio",               audio,
            "--outfile",             out_path,
            "--face_det_batch_size", str(self.cfg.wav2lip_face_det_batch),
            "--wav2lip_batch_size",  str(self.cfg.wav2lip_batch),
            "--resize_factor",       str(self.cfg.wav2lip_resize_factor),
            "--pads", str(top), str(bot), str(left), str(right),
        ]
        if not self.cfg.use_gpu or self.cfg.wav2lip_nosmooth:
            cmd.append("--nosmooth")
        r = subprocess.run(cmd, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Wav2Lip failed (code {r.returncode})")
        return out_path

    def _sadtalker(self, video: str, audio: str,
                   out_path: str) -> str:
        portrait = self._best_face_frame(video)
        result_dir = Path(self.cfg.tmp_dir) / "sadtalker_out"
        result_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, str(SADTALKER_DIR / "inference.py"),
            "--driven_audio",   audio,
            "--source_image",   portrait,
            "--result_dir",     str(result_dir),
            "--checkpoint_dir", self.cfg.sadtalker_checkpoint_dir,
            "--size",           str(self.cfg.sadtalker_size),
            "--preprocess",     "full",
        ]
        if self.cfg.sadtalker_still:
            cmd.append("--still")
        r = subprocess.run(cmd, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"SadTalker failed (code {r.returncode})")
        generated = sorted(result_dir.glob("*.mp4"))
        if not generated:
            raise RuntimeError("SadTalker produced no output")
        shutil.move(str(generated[-1]), out_path)
        return out_path

    def _sadtalker_ready(self) -> bool:
    """Check SadTalker has its required checkpoint before trying to run."""
    required = Path(self.cfg.sadtalker_checkpoint_dir) / "SadTalker_V0.0.2_256.safetensors"
    return required.exists()

    def _audio_swap(self, video: str, audio: str,
                    out_path: str) -> str:
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", video, "-i", audio,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac",
             "-shortest", out_path],
            check=True, capture_output=True)
        return out_path

    def _best_face_frame(self, video_path: str) -> str:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(video_path)
        best_frame, best_score = None, 0
        for i in range(300):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, 1.1, 4, minSize=(60, 60))
                score = (sum(int(w)*int(h)
                             for (_, _, w, h) in faces)
                         if len(faces) else 0)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
        cap.release()
        if best_frame is None:
            cap = cv2.VideoCapture(video_path)
            _, best_frame = cap.read()
            cap.release()
        portrait = safe_tmp_path(self.cfg.tmp_dir, "portrait.jpg")
        cv2.imwrite(portrait, best_frame)
        return portrait
