"""
Vrite - Audio Post-Processor
Silence trim, noise gate, EQ match, tempo adjust, loudness normalise.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("vrite.audio_post")


class AudioPostProcessor:
    def __init__(self, target_lufs: float = -14.0,
                 noise_gate_db: float = -50.0,
                 tempo_factor: float = 1.0,
                 sample_rate: int = 48000):
        self.target_lufs   = target_lufs
        self.noise_gate_db = noise_gate_db
        self.tempo_factor  = tempo_factor
        self.sample_rate   = sample_rate

    def process(self, input_path: str, output_path: str,
                reference_audio: str = None) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            trimmed = str(t / "01.wav")
            self._trim_silence(input_path, trimmed)
            gated = str(t / "02.wav")
            self._noise_gate(trimmed, gated)
            if reference_audio and Path(reference_audio).exists():
                try:
                    eq = str(t / "03.wav")
                    self._eq_match(gated, reference_audio, eq)
                    gated = eq
                except Exception as exc:
                    log.warning("EQ match skipped: %s", exc)
            temped = str(t / "04.wav")
            self._adjust_tempo(gated, temped)
            self._normalise(temped, output_path)
        log.info("Audio post-processing -> %s", output_path)
        return output_path

    def _trim_silence(self, src: str, dst: str) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-af",
             (f"silenceremove=start_periods=1:start_duration=0.1"
              f":start_threshold={self.noise_gate_db}dB"
              f":stop_periods=-1:stop_duration=0.3"
              f":stop_threshold={self.noise_gate_db}dB"),
             dst],
            check=True, capture_output=True)

    def _noise_gate(self, src: str, dst: str) -> None:
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        thr = 10 ** (self.noise_gate_db / 20.0)
        fs = int(sr * 0.02)
        for s in range(0, len(audio), fs):
            frame = audio[s:s + fs]
            if len(frame) and float(np.sqrt(np.mean(frame**2))) < thr:
                audio[s:s + fs] *= 0.05
        sf.write(dst, audio, sr, subtype="PCM_16")

    def _eq_match(self, src: str, ref: str, dst: str) -> None:
        from scipy.signal import fftconvolve

        def spectrum(p, n=2048):
            a, _ = sf.read(p, dtype="float32", always_2d=True)
            m = a.mean(axis=1)
            fs = [np.fft.rfft(m[i:i+n] * np.hanning(n), n=n)
                  for i in range(0, len(m)-n, n//2)]
            return (np.array([np.abs(f) for f in fs]).mean(axis=0) + 1e-10
                    if fs else np.ones(n//2+1))

        n = 2048
        gain = np.clip(spectrum(ref, n) / spectrum(src, n), 0.25, 4.0)
        cep = np.fft.irfft(np.log(gain))
        mc = np.zeros_like(cep)
        mc[0] = cep[0]
        mc[1:n//2] = 2 * cep[1:n//2]
        fir = np.real(np.fft.irfft(np.exp(np.fft.rfft(mc))))[:64]
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        filtered = np.stack(
            [fftconvolve(audio[:, c], fir, mode="same")
             for c in range(audio.shape[1])], axis=1)
        peak = np.max(np.abs(filtered)) + 1e-10
        if peak > 0.98:
            filtered /= peak * 1.02
        sf.write(dst, filtered.astype(np.float32), sr, subtype="PCM_16")

    def _adjust_tempo(self, src: str, dst: str) -> None:
        if abs(self.tempo_factor - 1.0) < 0.01:
            subprocess.run(["ffmpeg", "-y", "-i", src, "-c", "copy", dst],
                           check=True, capture_output=True)
            return
        f = max(0.5, min(2.0, self.tempo_factor))
        subprocess.run(["ffmpeg", "-y", "-i", src,
                        "-af", f"atempo={f:.3f}", dst],
                       check=True, capture_output=True)

    def _normalise(self, src: str, dst: str) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-af", (f"loudnorm=I={self.target_lufs}:TP=-1.5:LRA=11,"
                     f"aresample={self.sample_rate}"),
             "-ar", str(self.sample_rate),
             "-ac", "2", "-acodec", "pcm_s16le", dst],
            check=True, capture_output=True)
