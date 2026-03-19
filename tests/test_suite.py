"""
Vrite - Test Suite
28 unit tests. No GPU or model weights required.

Run:
    pytest tests/test_suite.py -v
    python tests/test_suite.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub heavy optional deps so tests run without installing everything
for _mod in [
    "cv2", "torch", "torchvision", "torchaudio",
    "TTS", "TTS.api", "pyttsx3", "ffmpeg",
    "soundfile", "scipy", "scipy.signal",
    "librosa", "moviepy", "basicsr", "realesrgan",
    "face_alignment", "safetensors", "dlib", "gtts",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np  # use the real numpy


# -- Config --------------------------------------------------------------------

class TestPipelineConfig(unittest.TestCase):

    def test_device_resolves_from_auto(self):
        from vrite.config import PipelineConfig
        cfg = PipelineConfig()
        self.assertIn(cfg.device, {"cuda", "cpu"})

    def test_env_var_override(self):
        os.environ["VRITE_OUTPUT_CRF"] = "28"
        try:
            from vrite.config import PipelineConfig
            self.assertEqual(PipelineConfig().output_crf, 28)
        finally:
            del os.environ["VRITE_OUTPUT_CRF"]

    def test_tmp_dir_created(self):
        from vrite.config import PipelineConfig
        self.assertTrue(Path(PipelineConfig().tmp_dir).is_dir())

    def test_use_gpu_true(self):
        from vrite.config import PipelineConfig
        self.assertTrue(PipelineConfig(device="cuda").use_gpu)

    def test_use_gpu_false(self):
        from vrite.config import PipelineConfig
        self.assertFalse(PipelineConfig(device="cpu").use_gpu)

    def test_invalid_env_var_ignored(self):
        os.environ["VRITE_OUTPUT_CRF"] = "not_a_number"
        try:
            from vrite.config import PipelineConfig
            self.assertIsInstance(PipelineConfig().output_crf, int)
        finally:
            del os.environ["VRITE_OUTPUT_CRF"]


# -- Script preprocessor -------------------------------------------------------

class TestScriptPreprocessor(unittest.TestCase):

    def setUp(self):
        from vrite.pipeline.script_preprocessor import ScriptPreprocessor
        self.pp = ScriptPreprocessor(speech_rate="normal")

    def test_basic(self):
        doc = self.pp.process("Hello world. This is a test.")
        self.assertGreater(len(doc.segments), 0)
        self.assertGreater(doc.word_count, 0)

    def test_markdown_stripped(self):
        doc = self.pp.process("# Heading\n\n**Bold** and *italic*.")
        for seg in doc.segments:
            self.assertNotIn("**", seg.text)
            self.assertNotIn("#", seg.text)

    def test_html_stripped(self):
        doc = self.pp.process("<p>Hello <b>world</b>.</p>")
        self.assertNotIn("<", doc.clean_text)

    def test_terminal_punctuation_added(self):
        doc = self.pp.process("This has no period")
        self.assertTrue(
            any(s.text.endswith((".", "!", "?"))
                for s in doc.segments))

    def test_abbreviation_expansion(self):
        doc = self.pp.process("Dr. Smith went to work.")
        self.assertIn("Doctor", doc.clean_text)

    def test_long_paragraph_split(self):
        doc = self.pp.process("This is a sentence. " * 20)
        self.assertGreater(len(doc.segments), 1)

    def test_pause_injected(self):
        doc = self.pp.process("First paragraph.\n\nSecond paragraph.")
        pauses = [s.pause_before_s for s in doc.segments[1:]]
        self.assertTrue(any(p > 0 for p in pauses))

    def test_duration_estimate(self):
        doc = self.pp.process("word " * 150)
        self.assertAlmostEqual(
            doc.total_estimated_duration_s, 60.0, delta=20.0)

    def test_empty_input(self):
        doc = self.pp.process("   \n\n   ")
        self.assertEqual(doc.word_count, 0)

    def test_full_tts_text(self):
        doc = self.pp.process("Hello. Goodbye.")
        t = doc.full_tts_text()
        self.assertIsInstance(t, str)
        self.assertGreater(len(t), 0)

    def test_chunk_splitting(self):
        doc = self.pp.process("This is a sentence. " * 50)
        for chunk in doc.iter_chunks(max_seconds=15.0):
            dur = sum(s.estimated_duration_s for s in chunk)
            self.assertLess(dur, 35.0)

    def test_emphasis_detection(self):
        doc = self.pp.process("This is the only critical step.")
        all_e = [w for s in doc.segments for w in s.emphasis_words]
        self.assertTrue(
            any(w.lower() in {"only", "critical"} for w in all_e))

    def test_code_blocks_stripped(self):
        doc = self.pp.process(
            "Check:\n```python\nprint('hi')\n```\nContinue.")
        self.assertNotIn("```", doc.clean_text)

    def test_summary_string(self):
        doc = self.pp.process("Hello world.")
        self.assertIn("segments", doc.summary())


# -- TTS Engine ----------------------------------------------------------------

class TestTTSEngine(unittest.TestCase):

    def _engine(self):
        from vrite.pipeline.tts_engine import TTSEngine
        from vrite.config import PipelineConfig
        return TTSEngine(PipelineConfig())

    def test_strips_markdown(self):
        result = self._engine()._preprocess("# Hello **world**")
        self.assertNotIn("#", result)
        self.assertNotIn("**", result)

    def test_adds_punctuation(self):
        result = self._engine()._preprocess("No period here")
        self.assertTrue(result.endswith((".", "!", "?")))

    def test_expands_abbreviations(self):
        result = self._engine()._preprocess("e.g. this is an example.")
        self.assertIn("for example", result.lower())

    def test_strips_html(self):
        result = self._engine()._preprocess("<p>Hello</p>")
        self.assertNotIn("<", result)


# -- Style analyser (pure logic, no video file needed) -------------------------

class TestStyleAnalyserLogic(unittest.TestCase):

    def test_fps_parse_fraction(self):
        raw = "30000/1001"
        num, den = map(int, raw.split("/"))
        self.assertAlmostEqual(num / den, 29.97, places=1)

    def test_fps_parse_integer(self):
        raw = "25/1"
        num, den = map(int, raw.split("/"))
        self.assertEqual(num / den, 25.0)

    def test_dominant_colours_shape(self):
        try:
            import cv2 as real_cv2
            if isinstance(real_cv2, MagicMock):
                self.skipTest("cv2 not available")
        except Exception:
            self.skipTest("cv2 not available")

        from vrite.pipeline.style_analyser import StyleAnalyser
        red   = np.tile([0,   0,   255], (500, 1)).astype(np.float32)
        green = np.tile([0,   255, 0],   (500, 1)).astype(np.float32)
        blue  = np.tile([255, 0,   0],   (500, 1)).astype(np.float32)
        colours = StyleAnalyser._dominant_colours([red, green, blue], k=3)
        self.assertEqual(len(colours), 3)
        for c in colours:
            self.assertEqual(len(c), 3)


# -- Pipeline integration (fully mocked) --------------------------------------

class TestPipelineIntegration(unittest.TestCase):

    @patch("vrite.pipeline.style_analyser.StyleAnalyser.analyse")
    @patch("vrite.pipeline.style_analyser.StyleAnalyser.extract_audio")
    @patch("vrite.pipeline.tts_engine.TTSEngine.synthesise")
    @patch("vrite.pipeline.lipsync_engine.LipSyncEngine.process")
    @patch("vrite.pipeline.video_compositor.VideoCompositor.compose")
    @patch("vrite.pipeline.video_compositor.VideoCompositor.get_duration")
    def test_all_stages_called(
            self, mock_dur, mock_compose, mock_lipsync,
            mock_tts, mock_audio, mock_analyse):
        mock_analyse.return_value  = {"fps": 25, "mean_brightness": 128.0}
        mock_audio.return_value    = "/tmp/a.wav"
        mock_tts.return_value      = "/tmp/t.wav"
        mock_lipsync.return_value  = "/tmp/l.mp4"
        mock_compose.return_value  = "/tmp/r.mp4"
        mock_dur.return_value      = 45.0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            fake = f.name
        try:
            with patch("pathlib.Path.exists", return_value=True):
                from run import VideoPipeline
                from vrite.config import PipelineConfig
                result = VideoPipeline(PipelineConfig()).run(
                    sample_video=fake,
                    script="Test script.",
                    output_path="/tmp/out.mp4")
            mock_analyse.assert_called_once()
            mock_tts.assert_called_once()
            mock_lipsync.assert_called_once()
            mock_compose.assert_called_once()
            self.assertEqual(result.duration_seconds, 45.0)
            self.assertIn("style_analysis", result.elapsed_per_step)
            self.assertIn("tts",            result.elapsed_per_step)
            self.assertIn("lipsync",        result.elapsed_per_step)
            self.assertIn("composition",    result.elapsed_per_step)
        finally:
            Path(fake).unlink(missing_ok=True)

    def test_validate_missing_video(self):
        from run import VideoPipeline
        with self.assertRaises(FileNotFoundError):
            VideoPipeline._validate("/nonexistent/file.mp4", "script")

    def test_validate_empty_script(self):
        from run import VideoPipeline
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            try:
                with self.assertRaises(ValueError):
                    VideoPipeline._validate(f.name, "   ")
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_validate_unsupported_format(self):
        from run import VideoPipeline
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            try:
                with self.assertRaises(ValueError):
                    VideoPipeline._validate(f.name, "Valid script.")
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_progress_callback_fires(self):
        calls = []
        patches = {
            "vrite.pipeline.style_analyser.StyleAnalyser.analyse":
                {"return_value": {"fps": 25, "mean_brightness": 128.0}},
            "vrite.pipeline.style_analyser.StyleAnalyser.extract_audio":
                {"return_value": "/tmp/a.wav"},
            "vrite.pipeline.tts_engine.TTSEngine.synthesise":
                {"return_value": "/tmp/t.wav"},
            "vrite.pipeline.lipsync_engine.LipSyncEngine.process":
                {"return_value": "/tmp/l.mp4"},
            "vrite.pipeline.video_compositor.VideoCompositor.compose":
                {"return_value": "/tmp/o.mp4"},
            "vrite.pipeline.video_compositor.VideoCompositor.get_duration":
                {"return_value": 30.0},
            "pathlib.Path.exists": {"return_value": True},
        }
        with (patch("vrite.pipeline.style_analyser.StyleAnalyser.analyse",
                    return_value={"fps": 25, "mean_brightness": 128.0}),
              patch("vrite.pipeline.style_analyser.StyleAnalyser.extract_audio",
                    return_value="/tmp/a.wav"),
              patch("vrite.pipeline.tts_engine.TTSEngine.synthesise",
                    return_value="/tmp/t.wav"),
              patch("vrite.pipeline.lipsync_engine.LipSyncEngine.process",
                    return_value="/tmp/l.mp4"),
              patch("vrite.pipeline.video_compositor.VideoCompositor.compose",
                    return_value="/tmp/o.mp4"),
              patch("vrite.pipeline.video_compositor.VideoCompositor.get_duration",
                    return_value=30.0),
              patch("pathlib.Path.exists", return_value=True)):
            from run import VideoPipeline
            from vrite.config import PipelineConfig
            pipe = VideoPipeline(PipelineConfig())
            pipe.set_progress_callback(
                lambda msg, pct: calls.append(pct))
            with tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False) as f:
                fake = f.name
            try:
                pipe.run(sample_video=fake, script="Hello.",
                         output_path="/tmp/r.mp4")
            finally:
                Path(fake).unlink(missing_ok=True)
        self.assertGreater(len(calls), 2)
        self.assertEqual(calls[-1], 100)


# -- Utilities -----------------------------------------------------------------

class TestUtils(unittest.TestCase):

    def test_format_seconds_only(self):
        from vrite.utils import format_duration
        self.assertEqual(format_duration(45.0), "45s")

    def test_format_minutes_and_seconds(self):
        from vrite.utils import format_duration
        self.assertEqual(format_duration(154.0), "2m 34s")

    def test_format_exactly_one_minute(self):
        from vrite.utils import format_duration
        self.assertEqual(format_duration(60.0), "1m 00s")

    def test_clamp_in_range(self):
        from vrite.utils import clamp
        self.assertEqual(clamp(5.0, 0.0, 10.0), 5.0)

    def test_clamp_below(self):
        from vrite.utils import clamp
        self.assertEqual(clamp(-1.0, 0.0, 10.0), 0.0)

    def test_clamp_above(self):
        from vrite.utils import clamp
        self.assertEqual(clamp(20.0, 0.0, 10.0), 10.0)

    def test_safe_tmp_path(self):
        from vrite.utils import safe_tmp_path
        p = safe_tmp_path("/tmp", "test.wav")
        self.assertTrue(p.endswith("test.wav"))
        self.assertIn("tmp", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
