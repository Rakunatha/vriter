"""
Vrite - Video Compositor
Colour grading, fades, audio mux, H.264/AAC export.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vrite.config import PipelineConfig
from vrite.utils import safe_tmp_path

log = logging.getLogger("vrite.compositor")


class VideoCompositor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def compose(self, video_path: str, audio_path: str,
                style_meta: dict[str, Any],
                output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.cfg.colour_grade:
            graded = safe_tmp_path(self.cfg.tmp_dir, "graded.mp4")
            log.info("Colour grading ...")
            self._grade(video_path, style_meta, graded)
        else:
            graded = video_path

        log.info("Encoding -> %s", output_path)
        self._encode(graded, audio_path, output_path)
        return output_path

    def get_duration(self, path: str) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-print_format", "json", "-show_format", path],
            capture_output=True, text=True)
        if r.returncode != 0:
            return 0.0
        return float(
            json.loads(r.stdout).get("format", {}).get("duration", 0.0))

    def _grade(self, video_path: str,
               style_meta: dict[str, Any],
               out_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or style_meta.get("fps", 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        target_b = style_meta.get("mean_brightness", 128.0)
        target_c = style_meta.get("contrast", 40.0)
        max_frames = int(fps * self.cfg.max_duration_seconds)
        idx = 0
        while idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(self._correct(frame, target_b, target_c))
            idx += 1
        cap.release()
        writer.release()

    @staticmethod
    def _correct(frame: np.ndarray,
                 target_b: float,
                 target_c: float) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur = float(gray.mean()) + 1e-6
        alpha = float(np.clip(
            1.0 + 0.15 * (target_b / cur - 1.0), 0.80, 1.30))
        beta = float(np.clip(0.4 * (target_b - cur), -25.0, 25.0))
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def _encode(self, video_path: str, audio_path: str,
                output_path: str) -> None:
        dur_v = self.get_duration(video_path)
        dur_a = self.get_duration(audio_path)
        clip_len = min(dur_v, dur_a, self.cfg.max_duration_seconds)
        fade = 0.5

        vf = []
        if self.cfg.add_outro_fade and clip_len > fade:
            vf.append(
                f"fade=t=out:st={clip_len - fade:.2f}:d={fade:.2f}")
        if self.cfg.output_resolution:
            w, h = self.cfg.output_resolution
            vf.append(f"scale={w}:{h}")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(clip_len),
            "-c:v", self.cfg.output_codec,
            "-crf", str(self.cfg.output_crf),
            "-preset", "medium",
            "-c:a", self.cfg.output_audio_codec,
            "-b:a", self.cfg.output_audio_bitrate,
            "-movflags", "+faststart",
        ]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd.append(output_path)

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{r.stderr[-2000:]}")
