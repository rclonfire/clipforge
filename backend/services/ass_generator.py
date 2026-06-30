"""
ASS subtitle file generator for ClipForge — Phase 03-export-engine.

Produces Advanced SubStation Alpha (.ass) subtitle files from word-level
timestamp lists. Each word gets its own Dialogue line for pop-on/pop-off
TikTok-style captions.

Interface:
    generate_ass(words: list[dict], output_path: str) -> str

Key behaviors:
- One Dialogue line per word (pop-on style)
- Words are uppercased (TikTok/Shorts aesthetic)
- Font: Montserrat 80pt, white text, bottom-center aligned (Alignment 2)
- Resolution: 1080x1920 (9:16 vertical video)
- MarginV: 120 (keeps captions above bottom safe zone)
- Timestamps formatted as H:MM:SS.cc (centiseconds)
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# ASS header constant
# ---------------------------------------------------------------------------
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,80,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def _fmt_time(seconds: float) -> str:
    """
    Format seconds as H:MM:SS.cc (centiseconds, 2 decimal places).

    Examples:
        0.0   -> "0:00:00.00"
        1.5   -> "0:00:01.50"
        60.0  -> "0:01:00.00"
        3600  -> "1:00:00.00"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_ass(words: list[dict], output_path: str) -> str:
    """
    Generate an ASS subtitle file with one Dialogue line per word.

    Args:
        words: List of word dicts with keys {word: str, start: float, end: float}.
               Timestamps should be relative to the clip start (seconds).
        output_path: Absolute path where the .ass file should be written.

    Returns:
        output_path — the same string passed in, on success.
    """
    lines = [ASS_HEADER]

    for w in words:
        start = _fmt_time(float(w["start"]))
        end = _fmt_time(float(w["end"]))
        text = w["word"].strip().upper()
        # Dialogue format: Layer, Start, End, Style, Name, ML, MR, MV, Effect, Text
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    content = "\n".join(lines) + "\n"
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path
