"""
Vrite - Script Preprocessor
Cleans, structures, and estimates timing for TTS input.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

WPM = {"slow": 110, "normal": 150, "fast": 190}

PAUSE_MAP = {
    ".": 0.55, "!": 0.55, "?": 0.55,
    ",": 0.20, ";": 0.35, ":": 0.30,
    "\n\n": 0.70,
}

ABBREV = {
    r"\bDr\.": "Doctor", r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus", r"\bMs\.": "Miss",
    r"\betc\.": "etcetera", r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is", r"\bvs\.": "versus",
}


@dataclass
class ScriptSegment:
    text: str
    raw_text: str
    estimated_duration_s: float = 0.0
    pause_before_s: float = 0.0
    segment_index: int = 0
    emphasis_words: list = field(default_factory=list)


@dataclass
class ScriptDoc:
    segments: list
    raw_text: str
    clean_text: str
    total_estimated_duration_s: float
    word_count: int
    language: str = "en"

    def full_tts_text(self) -> str:
        parts = []
        for seg in self.segments:
            if seg.pause_before_s >= 0.5:
                parts.append("..." * max(1, int(seg.pause_before_s / 0.4)))
            parts.append(seg.text)
        return " ".join(parts)

    def iter_chunks(self, max_seconds: float = 30.0) -> Iterator:
        chunk, dur = [], 0.0
        for seg in self.segments:
            if chunk and dur + seg.estimated_duration_s > max_seconds:
                yield chunk
                chunk, dur = [], 0.0
            chunk.append(seg)
            dur += seg.estimated_duration_s
        if chunk:
            yield chunk

    def summary(self) -> str:
        return (f"ScriptDoc: {len(self.segments)} segments, "
                f"{self.word_count} words, "
                f"~{self.total_estimated_duration_s:.0f}s")


class ScriptPreprocessor:
    def __init__(self, speech_rate: str = "normal",
                 language: str = "en",
                 max_segment_words: int = 60):
        self.wpm = WPM.get(speech_rate, WPM["normal"])
        self.language = language
        self.max_segment_words = max_segment_words

    def process(self, raw_text: str) -> ScriptDoc:
        text = self._normalise(raw_text)
        text = self._strip_markup(text)
        text = self._expand_abbrev(text)
        paras = self._split_paragraphs(text)
        segments = []
        for para in paras:
            segments.extend(
                self._para_to_segments(para, base_index=len(segments)))
        segments = self._inject_pauses(segments)
        for seg in segments:
            seg.emphasis_words = self._emphasis(seg.text)
            seg.estimated_duration_s = self._duration(seg.text)
        total = sum(s.estimated_duration_s + s.pause_before_s
                    for s in segments)
        words = len(re.findall(r"\b\w+\b", text))
        clean = " ".join(s.text for s in segments)
        return ScriptDoc(
            segments=segments, raw_text=raw_text,
            clean_text=clean, total_estimated_duration_s=total,
            word_count=words, language=self.language)

    @staticmethod
    def _normalise(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        for src, dst in [
            ("\u2018", "'"), ("\u2019", "'"),
            ("\u201c", '"'), ("\u201d", '"'),
            ("\u2013", "-"), ("\u2014", "--"),
            ("\u2026", "..."), ("\u00a0", " "),
        ]:
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _strip_markup(text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`\n]+`", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[*_~]{1,3}", "", text)
        text = re.sub(r"^\s*>+\s*", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _expand_abbrev(text: str) -> str:
        for pat, rep in ABBREV.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _split_paragraphs(text: str) -> list:
        return [re.sub(r"\s+", " ", p).strip()
                for p in re.split(r"\n{2,}", text) if p.strip()]

    def _para_to_segments(self, para: str,
                           base_index: int = 0) -> list:
        try:
            import nltk
            try:
                sents = nltk.sent_tokenize(para)
            except LookupError:
                nltk.download("punkt", quiet=True)
                sents = nltk.sent_tokenize(para)
        except ImportError:
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', para)
            sents = [s.strip() for s in sents if s.strip()]

        segments, cur_w, cur_s = [], [], []

        def flush():
            if cur_s:
                t = re.sub(r"\s+", " ", " ".join(cur_s)).strip()
                if t and t[-1] not in ".!?":
                    t += "."
                segments.append(ScriptSegment(
                    text=t, raw_text=" ".join(cur_s),
                    segment_index=base_index + len(segments)))
                cur_w.clear()
                cur_s.clear()

        for sent in sents:
            words = sent.split()
            if (cur_w and
                    len(cur_w) + len(words) > self.max_segment_words):
                flush()
            cur_w.extend(words)
            cur_s.append(sent)
        flush()
        return segments

    @staticmethod
    def _inject_pauses(segments: list) -> list:
        for i, seg in enumerate(segments):
            if i == 0:
                continue
            last = segments[i - 1].text.rstrip()
            seg.pause_before_s = PAUSE_MAP.get(
                last[-1] if last else "", 0.0)
        return segments

    @staticmethod
    def _emphasis(text: str) -> list:
        pat = (r"\b(never|always|every|must|critical|key|important|"
               r"essential|only|best|worst|first|last|new|free|now)\b")
        return list(set(re.findall(pat, text, flags=re.IGNORECASE)))

    def _duration(self, text: str) -> float:
        return (len(re.findall(r"\b\w+\b", text)) / self.wpm) * 60.0
