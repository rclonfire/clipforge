"""
Unit tests for caption_aligner.py.

whisperx is mocked at the module level so these tests run without
the real whisperx package installed.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inject a minimal whisperx stub into sys.modules BEFORE importing the module
# so that `import whisperx` in caption_aligner.py resolves to the stub.
# ---------------------------------------------------------------------------
def _make_whisperx_stub():
    stub = types.ModuleType("whisperx")
    stub.load_audio = MagicMock(return_value="audio_array")
    stub.load_align_model = MagicMock(return_value=(MagicMock(), {"language": "en"}))
    stub.align = MagicMock(return_value={
        "word_segments": [
            {"word": "Hello", "start": 5.0, "end": 5.3},
            {"word": "world", "start": 5.5, "end": 5.9},
            {"word": "outside", "start": 20.0, "end": 20.4},  # outside clip window
        ]
    })
    sys.modules["whisperx"] = stub
    return stub


_WHISPERX_STUB = _make_whisperx_stub()


from backend.services.caption_aligner import (  # noqa: E402
    align_words_for_clip,
    _fallback_to_faster_whisper_words,
    _align_model_cache,
)


class TestAlignWordsForClipReturnFormat(unittest.TestCase):
    """align_words_for_clip returns list of dicts with keys word, start, end (all floats relative to clip start)."""

    def setUp(self):
        _WHISPERX_STUB.align.return_value = {
            "word_segments": [
                {"word": "Hello", "start": 5.0, "end": 5.3},
                {"word": "world", "start": 5.5, "end": 5.9},
            ]
        }
        _WHISPERX_STUB.load_audio.return_value = "audio_array"
        _WHISPERX_STUB.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def test_returns_list_of_dicts(self):
        result = align_words_for_clip("audio.wav", [], clip_start=5.0, clip_end=10.0)
        self.assertIsInstance(result, list)

    def test_each_item_has_word_start_end_keys(self):
        result = align_words_for_clip("audio.wav", [], clip_start=5.0, clip_end=10.0)
        for item in result:
            self.assertIn("word", item)
            self.assertIn("start", item)
            self.assertIn("end", item)

    def test_timestamps_are_clip_relative(self):
        """start/end should be offset by -clip_start."""
        result = align_words_for_clip("audio.wav", [], clip_start=5.0, clip_end=10.0)
        self.assertAlmostEqual(result[0]["start"], 0.0)   # 5.0 - 5.0
        self.assertAlmostEqual(result[0]["end"],   0.3)   # 5.3 - 5.0
        self.assertAlmostEqual(result[1]["start"], 0.5)   # 5.5 - 5.0
        self.assertAlmostEqual(result[1]["end"],   0.9)   # 5.9 - 5.0


class TestAlignWordsForClipFiltering(unittest.TestCase):
    """Words outside the clip window are filtered out."""

    def setUp(self):
        _WHISPERX_STUB.align.return_value = {
            "word_segments": [
                {"word": "before", "start": 1.0, "end": 1.5},   # before clip
                {"word": "in",     "start": 5.0, "end": 5.3},   # inside
                {"word": "after",  "start": 15.0, "end": 15.5}, # after clip
            ]
        }
        _WHISPERX_STUB.load_audio.return_value = "audio_array"
        _WHISPERX_STUB.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def test_words_outside_clip_window_excluded(self):
        result = align_words_for_clip("audio.wav", [], clip_start=5.0, clip_end=10.0)
        words = [r["word"] for r in result]
        self.assertIn("in", words)
        self.assertNotIn("before", words)
        self.assertNotIn("after", words)


class TestAlignWordsForClipFiltersUnaligned(unittest.TestCase):
    """Words missing 'start' key (unaligned) are filtered out."""

    def setUp(self):
        _WHISPERX_STUB.align.return_value = {
            "word_segments": [
                {"word": "good", "start": 5.0, "end": 5.3},
                {"word": "bad"},  # missing start/end
                {"word": "also_bad", "end": 5.5},  # missing start
            ]
        }
        _WHISPERX_STUB.load_audio.return_value = "audio_array"
        _WHISPERX_STUB.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def test_words_without_start_key_excluded(self):
        result = align_words_for_clip("audio.wav", [], clip_start=5.0, clip_end=10.0)
        words = [r["word"] for r in result]
        self.assertIn("good", words)
        self.assertNotIn("bad", words)
        self.assertNotIn("also_bad", words)


class TestAlignWordsForClipFallback(unittest.TestCase):
    """Falls back to faster-whisper word timestamps when WhisperX returns empty alignment."""

    def setUp(self):
        # Simulate WhisperX returning empty word_segments
        _WHISPERX_STUB.align.return_value = {"word_segments": []}
        _WHISPERX_STUB.load_audio.return_value = "audio_array"
        _WHISPERX_STUB.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def test_falls_back_to_faster_whisper_words(self):
        """When WhisperX returns empty, fallback should use original words list."""
        original_words = [
            {"word": "hello", "start": 5.0, "end": 5.3, "probability": 0.99},
            {"word": "world", "start": 5.5, "end": 5.9, "probability": 0.95},
            {"word": "outside", "start": 20.0, "end": 20.4, "probability": 0.9},
        ]
        result = align_words_for_clip("audio.wav", original_words, clip_start=5.0, clip_end=10.0)
        self.assertEqual(len(result), 2)
        words = [r["word"] for r in result]
        self.assertIn("hello", words)
        self.assertIn("world", words)
        self.assertNotIn("outside", words)

    def test_fallback_timestamps_are_clip_relative(self):
        original_words = [
            {"word": "hello", "start": 5.0, "end": 5.3, "probability": 0.99},
        ]
        result = align_words_for_clip("audio.wav", original_words, clip_start=5.0, clip_end=10.0)
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[0]["end"], 0.3)


class TestFallbackToFasterWhisperWords(unittest.TestCase):
    """_fallback_to_faster_whisper_words filters and offsets timestamps."""

    def test_filters_to_clip_window(self):
        words = [
            {"word": "a", "start": 1.0, "end": 1.5},
            {"word": "b", "start": 5.0, "end": 5.5},
            {"word": "c", "start": 12.0, "end": 12.5},
        ]
        result = _fallback_to_faster_whisper_words(words, clip_start=5.0, clip_end=10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["word"], "b")

    def test_offsets_timestamps_relative_to_clip_start(self):
        words = [{"word": "b", "start": 7.0, "end": 7.5}]
        result = _fallback_to_faster_whisper_words(words, clip_start=5.0, clip_end=10.0)
        self.assertAlmostEqual(result[0]["start"], 2.0)
        self.assertAlmostEqual(result[0]["end"], 2.5)


if __name__ == "__main__":
    unittest.main()
