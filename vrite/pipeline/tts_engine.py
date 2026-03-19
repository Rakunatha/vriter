"""
Vrite - TTS Engine
Fallback chain: Coqui XTTS-v2 -> gTTS -> pyttsx3
gTTS works on Render free tier with no model downloads.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from vrite.config import PipelineConfig
from vrite.utils import safe_tmp_path, trim_audio

log = logging.getLogger("vrite.tts")


class TTSEngine:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._coqui = None

    def synthesise(self, text: str,
                   reference_audio: str = None,
                   out_dir: str = None) -> str:
        out_dir = Path(out_dir or self.cfg.tmp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "tts_output.wav")
        clean = self._preprocess(text)
        log.info("Synthesising %d chars ...", len(clean))

        for engine in [
            lambda: self._try_coqui(clean, reference_audio, out_path),
            lambda: self._try_gtts(clean, out_path),
            lambda: self._try_pyttsx3(clean, out_path),
        ]:
            result = engine()
            if result:
                trimmed = out_path.replace(".wav", "_trim.wav")
                trim_audio(result, trimmed, self.cfg.max_duration_seconds)
                return trimmed

        raise RuntimeError(
            "All TTS engines failed. "
            "Install gTTS: pip install gTTS")

    # Engine 1: Coqui XTTS-v2 (best quality, needs ~2GB model)
    def _try_coqui(self, text: str,
                   reference_audio: str, out_path: str):
        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError:
            return None
        try:
            key = "clone" if reference_audio else "std"
            if not self._coqui or self._coqui[0] != key:
                model = (self.cfg.tts_model if reference_audio
                         else self.cfg.tts_fallback_model)
                self._coqui = (key, CoquiTTS(model).to(self.cfg.device))
            tts = self._coqui[1]
            if reference_audio and self._supports_cloning(tts):
                ref = self._clip_reference(reference_audio)
                tts.tts_to_file(
                    text=text, speaker_wav=ref,
                    language=self.cfg.tts_language,
                    file_path=out_path,
                    speed=self.cfg.tts_speed)
            else:
                tts.tts_to_file(text=text, file_path=out_path)
            log.info("Engine: Coqui XTTS-v2")
            return out_path
        except Exception as exc:
            log.warning("Coqui failed: %s", exc)
            return None

    @staticmethod
    def _supports_cloning(tts) -> bool:
        try:
            import inspect
            return "speaker_wav" in inspect.signature(
                tts.tts_to_file).parameters
        except Exception:
            return False

    def _clip_reference(self, audio_path: str) -> str:
        out = safe_tmp_path(self.cfg.tmp_dir, "ref_clip.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-t", str(self.cfg.tts_reference_clip_seconds),
             "-ac", "1", "-ar", "22050",
             "-acodec", "pcm_s16le", out],
            check=True, capture_output=True)
        return out

    # Engine 2: gTTS (Google, HTTP, no local model needed)
    def _try_gtts(self, text: str, out_path: str):
        try:
            from gtts import gTTS
        except ImportError:
            return None
        try:
            mp3 = out_path.replace(".wav", "_gtts.mp3")
            gTTS(text=text, lang=self.cfg.tts_language,
                 slow=False).save(mp3)
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3,
                 "-ac", "1", "-ar", "16000",
                 "-acodec", "pcm_s16le", out_path],
                check=True, capture_output=True)
            Path(mp3).unlink(missing_ok=True)
            log.info("Engine: gTTS")
            return out_path
        except Exception as exc:
            log.warning("gTTS failed: %s", exc)
            return None

    # Engine 3: pyttsx3 (fully offline system TTS)
    def _try_pyttsx3(self, text: str, out_path: str):
        try:
            import pyttsx3
        except ImportError:
            return None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", int(160 * self.cfg.tts_speed))
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            log.info("Engine: pyttsx3")
            return out_path
        except Exception as exc:
            log.warning("pyttsx3 failed: %s", exc)
            return None

    @staticmethod
    def _preprocess(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`]*`", " ", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"[*_~]{1,3}", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        abbrev = {
            r"\bDr\.": "Doctor", r"\bMr\.": "Mister",
            r"\bMrs\.": "Missus", r"\betc\.": "etcetera",
            r"\be\.g\.": "for example", r"\bi\.e\.": "that is",
            r"\bvs\.": "versus",
        }
        for pat, rep in abbrev.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text
