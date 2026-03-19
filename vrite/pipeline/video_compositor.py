"""
Vrite - Video Compositor
Uses ffmpeg filters instead of frame-by-frame OpenCV - 10-20x faster on CPU.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from vrite.config import PipelineConfig

log = logging.getLogger("vrite.compositor")


class VideoCompositor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def compose(self, video_path: str, audio_path: str,
                style_meta: dict[str, Any],
                output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Encoding final video -> %s", output_path)
        self._encode(video_path, audio_path, output_path, style_meta)
        return output_path

    def get_duration(self, path: str) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-print_format", "json",
             "-show_format", path],
            capture_output=True, text=True)
        if r.returncode != 0:
            return 0.0
        return float(
            json.loads(r.stdout).get("format", {}).get("duration", 0.0))

    def _encode(self, video_path: str, audio_path: str,
                output_path: str, style_meta: dict[str, Any]) -> None:

        dur_v = self.get_duration(video_path)
        dur_a = self.get_duration(audio_path)
        clip_len = min(dur_v, dur_a, self.cfg.max_duration_seconds)
        fade = 0.5

        vf_parts = []

        # Colour grading via ffmpeg eq filter (fast - no frame loop)
        if self.cfg.colour_grade:
            brightness = style_meta.get("mean_brightness", 128.0)
            # Convert 0-255 brightness to ffmpeg eq range (-1.0 to 1.0)
            eq_brightness = round((brightness - 128.0) / 255.0, 3)
            eq_brightness = max(-0.3, min(0.3, eq_brightness))
            if abs(eq_brightness) > 0.01:
                vf_parts.append(
                    f"eq=brightness={eq_brightness}:contrast=1.05")

        # Fade out
        if self.cfg.add_outro_fade and clip_len > fade:
            vf_parts.append(
                f"fade=t=out:st={clip_len - fade:.2f}:d={fade:.2f}")

        # Fade in
        if self.cfg.add_intro_fade:
            vf_parts.append(f"fade=t=in:st=0:d={fade:.2f}")

        # Scale if needed
        if self.cfg.output_resolution:
            w, h = self.cfg.output_resolution
            vf_parts.append(f"scale={w}:{h}")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(clip_len),
            "-c:v", self.cfg.output_codec,
            "-crf", str(self.cfg.output_crf),
            "-preset", "ultrafast",
            "-c:a", self.cfg.output_audio_codec,
            "-b:a", self.cfg.output_audio_bitrate,
            "-movflags", "+faststart",
        ]

        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        cmd.append(output_path)

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{r.stderr[-2000:]}")
