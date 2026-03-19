"""
Vrite - Main Entry Point
CLI and Python API for generating style-matched talking-head videos.

CLI usage:
    python run.py --sample-video ref.mp4 --script "Your script here"
    python run.py --sample-video ref.mp4 --script script.txt --output out.mp4

Python API:
    from run import VideoPipeline
    from vrite.config import PipelineConfig
    result = VideoPipeline().run(sample_video="ref.mp4", script="Hello")
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vrite.config import PipelineConfig
from vrite.pipeline.style_analyser import StyleAnalyser
from vrite.pipeline.tts_engine import TTSEngine
from vrite.pipeline.lipsync_engine import LipSyncEngine
from vrite.pipeline.video_compositor import VideoCompositor
from vrite.utils import setup_logging, format_duration

log = logging.getLogger("vrite.pipeline")


@dataclass
class PipelineResult:
    output_path: str
    duration_seconds: float
    style_meta: dict
    elapsed_total: float
    elapsed_per_step: dict


class VideoPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.cfg = config or PipelineConfig()
        self._progress_cb: Callable = None

    def set_progress_callback(self, cb: Callable) -> None:
        self._progress_cb = cb

    def _progress(self, msg: str, pct: int) -> None:
        log.info("[%3d%%] %s", pct, msg)
        if self._progress_cb:
            self._progress_cb(msg, pct)

    def run(self, sample_video: str, script: str,
            sample_audio: str = None,
            output_path: str = None) -> PipelineResult:
        t_total = time.perf_counter()
        elapsed = {}

        output_path = output_path or str(
            Path("outputs") / f"result_{int(time.time())}.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self._progress("Validating inputs ...", 2)
        self._validate(sample_video, script)

        # Step 1 - Style analysis
        self._progress("Analysing reference video ...", 8)
        t0 = time.perf_counter()
        analyser = StyleAnalyser(self.cfg)
        style_meta = analyser.analyse(sample_video)
        audio_ref = sample_audio or analyser.extract_audio(sample_video)
        elapsed["style_analysis"] = time.perf_counter() - t0

        # Step 2 - Text-to-speech
        self._progress("Synthesising speech ...", 22)
        t0 = time.perf_counter()
        tts_audio = TTSEngine(self.cfg).synthesise(
            text=script, reference_audio=audio_ref,
            out_dir=self.cfg.tmp_dir)
        elapsed["tts"] = time.perf_counter() - t0

        # Step 3 - Lip-sync
        self._progress("Running lip-sync ...", 48)
        t0 = time.perf_counter()
        lipsync_video = LipSyncEngine(self.cfg).process(
            source_video=sample_video, driven_audio=tts_audio,
            style_meta=style_meta, out_dir=self.cfg.tmp_dir)
        elapsed["lipsync"] = time.perf_counter() - t0

        # Step 4 - Compose
        self._progress("Compositing final video ...", 82)
        t0 = time.perf_counter()
        compositor = VideoCompositor(self.cfg)
        final = compositor.compose(
            video_path=lipsync_video, audio_path=tts_audio,
            style_meta=style_meta, output_path=output_path)
        elapsed["composition"] = time.perf_counter() - t0

        duration = compositor.get_duration(final)
        elapsed_total = time.perf_counter() - t_total
        self._progress(f"Done! Saved -> {final}", 100)
        log.info("Complete in %s | output: %s (%.1fs)",
                 format_duration(elapsed_total), final, duration)

        return PipelineResult(
            output_path=final,
            duration_seconds=duration,
            style_meta=style_meta,
            elapsed_total=elapsed_total,
            elapsed_per_step=elapsed)

    @staticmethod
    def _validate(sample_video: str, script: str) -> None:
        p = Path(sample_video)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {sample_video}")
        if p.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise ValueError(f"Unsupported format: {p.suffix}")
        if not script or not script.strip():
            raise ValueError("Script cannot be empty.")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="vrite",
        description="Vrite - generate a style-matched talking-head video")
    p.add_argument("--sample-video", required=True, metavar="PATH",
                   help="Reference video file (MP4/MOV/AVI)")
    p.add_argument("--script", required=True, metavar="TEXT_OR_FILE",
                   help="Script text or path to a .txt file")
    p.add_argument("--sample-audio", default=None, metavar="PATH",
                   help="Optional separate voice reference (WAV/MP3)")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Output MP4 path (default: outputs/result_<ts>.mp4)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--quality", default=20, type=int, metavar="CRF",
                   help="Video quality 0-51, lower=better (default: 20)")
    p.add_argument("--max-duration", default=120, type=int,
                   help="Max output seconds (default: 120)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(debug=args.verbose)

    script_text = args.script
    sp = Path(args.script)
    if sp.is_file() and sp.suffix in {".txt", ".md"}:
        script_text = sp.read_text(encoding="utf-8")

    cfg = PipelineConfig(
        device=args.device,
        output_crf=args.quality,
        max_duration_seconds=args.max_duration)

    try:
        result = VideoPipeline(cfg).run(
            sample_video=args.sample_video,
            script=script_text,
            sample_audio=args.sample_audio,
            output_path=args.output)
        print(f"\nVideo saved : {result.output_path}")
        print(f"Duration    : {result.duration_seconds:.1f}s")
        print(f"Total time  : {format_duration(result.elapsed_total)}")
        for step, secs in result.elapsed_per_step.items():
            print(f"  {step:<22s} {secs:5.1f}s")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
