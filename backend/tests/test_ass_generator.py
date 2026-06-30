"""
Unit tests for ass_generator.py.

Pure unit tests — no mocking needed. Verifies ASS header content,
Dialogue line count, timestamp format, and file output.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from backend.services.ass_generator import generate_ass, _fmt_time, ASS_HEADER


class TestASSHeader(unittest.TestCase):
    """generate_ass produces valid ASS header with required fields."""

    def test_header_contains_script_info(self):
        self.assertIn("[Script Info]", ASS_HEADER)

    def test_header_contains_playresx_1080(self):
        self.assertIn("PlayResX: 1080", ASS_HEADER)

    def test_header_contains_playresy_1920(self):
        self.assertIn("PlayResY: 1920", ASS_HEADER)

    def test_style_contains_montserrat(self):
        self.assertIn("Montserrat", ASS_HEADER)

    def test_style_contains_fontsize_80(self):
        # Style line format: Style: Default,Montserrat,80,...
        self.assertIn(",80,", ASS_HEADER)

    def test_style_contains_alignment_2(self):
        # Alignment value 2 appears in the style definition
        # Style: Default,Montserrat,80,...,2,60,60,120,1
        # We verify it's present in the style line
        style_line = [l for l in ASS_HEADER.splitlines() if l.startswith("Style:")]
        self.assertTrue(len(style_line) >= 1, "No Style: line found")
        # Alignment is the 19th field (0-indexed 18) in the style format
        fields = style_line[0].split(",")
        # Format: Name,Fontname,Fontsize,Primary,Secondary,Outline,Back,Bold,Italic,Underline,
        #         StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,...
        # Index:  0    1        2        3        4         5       6    7    8      9
        #         10       11     12     13      14     15           16      17     18
        alignment_idx = 18
        self.assertEqual(fields[alignment_idx].strip(), "2")

    def test_style_contains_marginv_120(self):
        style_line = [l for l in ASS_HEADER.splitlines() if l.startswith("Style:")]
        self.assertTrue(len(style_line) >= 1)
        # MarginV is index 21 in the style fields
        fields = style_line[0].split(",")
        # Format: ...,Alignment,MarginL,MarginR,MarginV,Encoding
        # Index:      18        19      20      21      22
        marginv_idx = 21
        self.assertEqual(fields[marginv_idx].strip(), "120")


class TestGenerateASSDialogueLines(unittest.TestCase):
    """generate_ass produces one Dialogue line per word."""

    def _make_words(self, n):
        return [{"word": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(n)]

    def test_one_dialogue_line_per_word(self):
        words = self._make_words(5)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass(words, path)
            content = open(path, encoding="utf-8").read()
            dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
            self.assertEqual(len(dialogue_lines), 5)
        finally:
            os.unlink(path)

    def test_zero_words_produces_no_dialogue_lines(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass([], path)
            content = open(path, encoding="utf-8").read()
            dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
            self.assertEqual(len(dialogue_lines), 0)
        finally:
            os.unlink(path)

    def test_dialogue_text_is_uppercase(self):
        words = [{"word": "hello", "start": 0.0, "end": 0.4}]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass(words, path)
            content = open(path, encoding="utf-8").read()
            dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
            # Last field in the Dialogue line is the text
            text_field = dialogue_lines[0].split(",,")[-1]
            self.assertEqual(text_field, "HELLO")
        finally:
            os.unlink(path)


class TestGenerateASSTimestampFormat(unittest.TestCase):
    """generate_ass formats timestamps as H:MM:SS.cc."""

    def test_fmt_time_zero(self):
        self.assertEqual(_fmt_time(0.0), "0:00:00.00")

    def test_fmt_time_one_and_half_seconds(self):
        self.assertEqual(_fmt_time(1.5), "0:00:01.50")

    def test_fmt_time_one_minute(self):
        self.assertEqual(_fmt_time(60.0), "0:01:00.00")

    def test_fmt_time_one_hour(self):
        self.assertEqual(_fmt_time(3600.0), "1:00:00.00")

    def test_fmt_time_complex(self):
        # 1h 2m 3.45s
        t = 3600 + 120 + 3.45
        self.assertEqual(_fmt_time(t), "1:02:03.45")

    def test_dialogue_line_uses_h_mm_ss_format(self):
        words = [{"word": "hi", "start": 1.5, "end": 2.0}]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass(words, path)
            content = open(path, encoding="utf-8").read()
            dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
            # Should contain the formatted timestamp
            self.assertIn("0:00:01.50", dialogue_lines[0])
            self.assertIn("0:00:02.00", dialogue_lines[0])
        finally:
            os.unlink(path)


class TestGenerateASSFileOutput(unittest.TestCase):
    """generate_ass writes file to output_path and returns that path."""

    def test_returns_output_path(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            result = generate_ass([], path)
            self.assertEqual(result, path)
        finally:
            os.unlink(path)

    def test_file_is_written(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass([{"word": "test", "start": 0.0, "end": 0.5}], path)
            self.assertTrue(os.path.exists(path))
            content = open(path, encoding="utf-8").read()
            self.assertGreater(len(content), 0)
        finally:
            os.unlink(path)

    def test_file_contains_script_info_header(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".ass", delete=False) as f:
            path = f.name
        try:
            generate_ass([], path)
            content = open(path, encoding="utf-8").read()
            self.assertIn("[Script Info]", content)
            self.assertIn("PlayResX: 1080", content)
            self.assertIn("PlayResY: 1920", content)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
