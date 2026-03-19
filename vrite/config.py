"""
Vrite - Pipeline Configuration
Override any setting via VRITE_* environment variables.
Example: VRITE_OUTPUT_CRF=22 python run.py ...
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Compute
    device: str = "auto"

    # I/O
    tmp_dir: str = field(
        default_factory=lambda: tempfile.mkdtemp(prefix="vrite_"))
    output_dir: str = "outputs"

    # Style analysis
    style_sample_every_n_frames: int = 30
    scene_threshold: float = 0.35
    fps_target: int = 25

    # TTS
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_fallback_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_language: str = "en"
    tts_speed: float = 1.0
    tts_reference_clip_seconds: float = 10.0

    # Wav2Lip
    wav2lip_checkpoint: str = "models/wav2lip_gan.pth"
    wav2lip_face_det_batch: int = 16
    wav2lip_batch: int = 128
    wav2lip_resize_factor: int = 1
    wav2lip_pads: tuple = (0, 10, 0, 0)
    wav2lip_nosmooth: bool = False

    # SadTalker
    sadtalker_checkpoint_dir: str = "models/sadtalker"
    sadtalker_size: int = 256
    sadtalker_still: bool = True

    # Video output
    output_codec: str = "libx264"
    output_crf: int = 20
    output_audio_codec: str = "aac"
    output_audio_bitrate: str = "192k"
    output_resolution: tuple = None
    max_duration_seconds: int = 120
    add_intro_fade: bool = True
    add_outro_fade: bool = True
    colour_grade: bool = True

    def __post_init__(self) -> None:
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        for key in self.__dataclass_fields__:
            env_val = os.environ.get(f"VRITE_{key.upper()}")
            if env_val is not None:
                expected = type(getattr(self, key))
                try:
                    setattr(self, key, expected(env_val))
                except (TypeError, ValueError):
                    pass

        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def use_gpu(self) -> bool:
        return self.device == "cuda"
