"""
Vrite - Style Analyser
Extracts visual and audio characteristics from a reference video.
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
from vrite.utils import require_ffmpeg, safe_tmp_path

log = logging.getLogger("vrite.analyser")


class StyleAnalyser:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        require_ffmpeg()

    def analyse(self, video_path: str) -> dict[str, Any]:
        log.info("Analysing: %s", video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)
        meta = {}
        meta.update(self._probe_container(video_path))
        meta.update(self._analyse_frames(video_path))
        return meta

    def extract_audio(self, video_path: str,
                      out_wav: str = None,
                      max_seconds: float = None) -> str:
        out_wav = out_wav or safe_tmp_path(self.cfg.tmp_dir, "ref_audio.wav")
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd.append(out_wav)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {r.stderr}")
        return out_wav

    def _probe_container(self, video_path: str) -> dict[str, Any]:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", video_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffprobe error: {r.stderr}")
        data = json.loads(r.stdout)
        vs = next((s for s in data.get("streams", [])
                   if s.get("codec_type") == "video"), {})
        raw_fps = vs.get("avg_frame_rate", "25/1")
        try:
            num, den = map(int, raw_fps.split("/"))
            fps = num / den if den else 25.0
        except Exception:
            fps = 25.0
        return {
            "fps": round(fps, 3),
            "width": vs.get("width", 512),
            "height": vs.get("height", 512),
            "duration": float(data.get("format", {}).get("duration", 60.0)),
            "codec": vs.get("codec_name", "h264"),
        }

    def _analyse_frames(self, video_path: str) -> dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = self.cfg.style_sample_every_n_frames
        prev_gray = None
        brightness, contrasts, motions, colour_buf, scene_cuts = \
            [], [], [], [], []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % n == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean()))
                contrasts.append(float(gray.std()))
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    motions.append(float(mag.mean()))
                    diff = float(np.abs(
                        gray.astype(float) - prev_gray.astype(float)
                    ).mean())
                    if diff > (255 * self.cfg.scene_threshold):
                        scene_cuts.append(frame_idx / fps)
                prev_gray = gray
                small = cv2.resize(frame, (64, 64))
                colour_buf.append(small.reshape(-1, 3))
            frame_idx += 1
        cap.release()
        return {
            "mean_brightness": float(np.mean(brightness)) if brightness else 128.0,
            "contrast": float(np.mean(contrasts)) if contrasts else 40.0,
            "mean_motion": float(np.mean(motions)) if motions else 1.0,
            "dominant_colours_bgr": (
                self._dominant_colours(colour_buf) if colour_buf else []),
            "scene_cuts": scene_cuts,
        }

    @staticmethod
    def _dominant_colours(colour_buf: list, k: int = 3) -> list:
        pixels = np.vstack(colour_buf).astype(np.float32)
        if len(pixels) > 8000:
            idx = np.random.choice(len(pixels), 8000, replace=False)
            pixels = pixels[idx]
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centres = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        order = np.argsort(-counts)
        return [centres[i].astype(int).tolist() for i in order]
