"""
Autonomous song identification for ClipForge violin / music covers.

The clips are instrumental (solo violin), so speech transcription yields no
lyrics and fingerprinting (Shazam-style) fails on covers. This service asks a
Gemini audio model to identify the song by its MELODY, with an explicit
confidence signal. The caption layer only names the song when confidence is
high enough (is_named); otherwise it falls back to a vibe-only caption — which
is exactly the "mix named and vibe-only" behavior the creator wants.

    identify_song(audio_path, ffmpeg_path) -> {song, artist, confidence, raw}
        confidence: "high" | "medium" | "low" | "none"
    is_named(result) -> bool   # True only when a song is named at high/medium confidence
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from backend.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Multimodal text model that accepts audio input (NOT the -image variant).
MODEL_ID = "gemini-2.5-flash"

_NAMED_CONFIDENCE = {"high", "medium"}

_PROMPT = (
    "This audio is a SOLO VIOLIN cover — instrumental, no vocals, and possibly a different "
    "arrangement, tempo, or key than the original. Identify the ORIGINAL song whose melody "
    "is being played.\n\n"
    "Be honest about certainty. Only give high or medium confidence if the melody is clearly "
    "recognizable as a specific, well-known song. If you are guessing, if it could be many "
    "songs, or if it's obscure or an original piece, use low or none and do NOT invent a title.\n\n"
    "Return ONLY a JSON object: "
    '{"song": "...", "artist": "...", "confidence": "high"|"medium"|"low"|"none"}. '
    "Use empty strings for song and artist when confidence is none."
)


def identify_song(audio_path: str, ffmpeg_path: str = "ffmpeg") -> dict:
    """Identify the song in an instrumental cover. Degrades to confidence 'none' on any failure."""
    result = {"song": "", "artist": "", "confidence": "none", "raw": ""}

    if not GEMINI_API_KEY:
        logger.info("song_identify: GEMINI_API_KEY not set — captions will be vibe-only")
        return result

    src = Path(audio_path)
    if not src.exists():
        logger.warning("song_identify: audio not found at %s", audio_path)
        return result

    clip = src
    try:
        clip = _extract_segment(src, ffmpeg_path)
    except Exception as exc:  # noqa: BLE001 — fall back to full audio
        logger.warning("song_identify: segment extraction failed (%s) — using full audio", exc)
        clip = src

    try:
        result = _ask_gemini(clip)
        logger.info(
            "song_identify: %r by %r (confidence=%s)",
            result.get("song"), result.get("artist"), result.get("confidence"),
        )
    except Exception as exc:  # noqa: BLE001 — never block the pipeline
        logger.warning("song_identify: Gemini call failed (%s) — vibe-only", exc)
    finally:
        if clip != src:
            try:
                Path(clip).unlink(missing_ok=True)
            except Exception:
                pass

    return result


def is_named(result: dict | None) -> bool:
    """True only when a song was identified at high or medium confidence."""
    if not result:
        return False
    return bool(result.get("song")) and result.get("confidence") in _NAMED_CONFIDENCE


def _extract_segment(src: Path, ffmpeg_path: str, start: float = 5.0, dur: float = 40.0) -> Path:
    """Extract a short mono segment as mp3 to keep the Gemini request small and focused."""
    out = Path(tempfile.gettempdir()) / f"songid_{src.stem}.mp3"
    cmd = [
        ffmpeg_path, "-y", "-ss", str(start), "-t", str(dur), "-i", str(src),
        "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-q:a", "5", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError((r.stderr or "")[:200])
    return out


def _ask_gemini(audio_file: Path) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    data = Path(audio_file).read_bytes()

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[_PROMPT, types.Part.from_bytes(data=data, mime_type="audio/mpeg")],
    )

    text = (getattr(response, "text", "") or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()

    data = json.loads(text)
    confidence = str(data.get("confidence", "none")).lower()
    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "low"
    return {
        "song": str(data.get("song", "")).strip(),
        "artist": str(data.get("artist", "")).strip(),
        "confidence": confidence,
        "raw": text[:300],
    }
