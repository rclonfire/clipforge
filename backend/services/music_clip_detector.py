"""
Music clip-detection brain for ClipForge.

The comedy brain (clip_detector.py) reads a speech transcript and scores
punchlines/reactions. That is useless for instrumental covers (no dialogue).
This brain judges the MUSIC: Gemini listens to the full performance and returns
the best segments (the recognizable hook, the climax, the impressive runs).
Timestamps are snapped to librosa onset_times so cuts land on phrase boundaries.

    looks_instrumental(words, duration) -> bool     # routing: speech vs music
    detect_music_clips(audio_path, signal_data, duration_seconds, ffmpeg_path) -> list[dict]

Output dicts match clip_detector.detect_clips so the rest of the pipeline
(preview extraction, export, post_prep) is unchanged.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from backend.config import GEMINI_API_KEY
from backend.prompts.music_clip_system import MUSIC_CLIP_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash"

MIN_CLIP = 20.0
MAX_CLIP = 60.0
TARGET_CLIP = 33.0
MAX_CLIPS = 8
_SNAP_WINDOW = 1.5          # snap start/end to an onset within this many seconds
_OVERLAP_GAP = 10.0         # treat clips starting within this many seconds as duplicates
# Below this words-per-second the video is treated as instrumental (speech is ~2-3 wps).
_SPEECH_WPS = 0.3


def looks_instrumental(words: list | None, duration: float) -> bool:
    """Route to the music brain when there's essentially no speech."""
    n = len(words or [])
    if duration and duration > 0:
        return (n / duration) < _SPEECH_WPS
    return n < 10


def detect_music_clips(
    audio_path: str,
    signal_data: dict,
    duration_seconds: float,
    ffmpeg_path: str = "ffmpeg",
) -> list[dict]:
    """
    Detect the best musical clip segments. Uses Gemini audio when a key is
    configured; otherwise falls back to the densest energy regions so a raw
    performance still cuts sensibly.
    """
    onsets = [float(t) for t in (signal_data or {}).get("onset_times", [])]

    raw: list[dict] = []
    if GEMINI_API_KEY:
        try:
            raw = _gemini_clips(audio_path, duration_seconds, ffmpeg_path)
        except Exception as exc:  # noqa: BLE001 — degrade to energy heuristic
            logger.warning("music_clip_detector: Gemini analysis failed (%s) — energy fallback", exc)

    if not raw:
        raw = _energy_fallback(signal_data or {}, duration_seconds)

    # Snap to phrase boundaries, enforce duration, dedup overlaps, cap.
    normalized = []
    for c in raw:
        n = _normalize(c, duration_seconds, onsets)
        if n:
            normalized.append(n)

    deduped = _dedup(normalized)
    logger.info("music_clip_detector: %d clip(s) after normalize+dedup", len(deduped))
    return deduped[:MAX_CLIPS]


# ---------------------------------------------------------------------------
# Gemini brain
# ---------------------------------------------------------------------------

def _gemini_clips(audio_path: str, duration_seconds: float, ffmpeg_path: str) -> list[dict]:
    from google import genai
    from google.genai import types

    clip_audio = _extract_mono_mp3(Path(audio_path), ffmpeg_path)
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        data = clip_audio.read_bytes()
        prompt = (
            f"{MUSIC_CLIP_SYSTEM_PROMPT}\n\nThe audio is about {int(duration_seconds)} seconds long."
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, types.Part.from_bytes(data=data, mime_type="audio/mpeg")],
        )
    finally:
        try:
            clip_audio.unlink(missing_ok=True)
        except Exception:
            pass

    text = (getattr(response, "text", "") or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()
    data = json.loads(text)
    return data.get("clips", []) if isinstance(data, dict) else []


def _extract_mono_mp3(src: Path, ffmpeg_path: str) -> Path:
    out = Path(tempfile.gettempdir()) / f"musicclip_{src.stem}.mp3"
    cmd = [
        ffmpeg_path, "-y", "-i", str(src),
        "-ac", "1", "-ar", "22050", "-c:a", "libmp3lame", "-q:a", "5", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError((r.stderr or "")[:200])
    return out


# ---------------------------------------------------------------------------
# Energy fallback (no Gemini)
# ---------------------------------------------------------------------------

def _energy_fallback(signal_data: dict, duration: float) -> list[dict]:
    """Pick clip windows around the densest energy regions from signal analysis."""
    peaks = sorted(float(p["time_seconds"]) for p in signal_data.get("energy_peaks", []) if "time_seconds" in p)
    if not peaks or duration <= 0:
        return _even_windows(duration)

    win = TARGET_CLIP
    step = win / 2.0
    candidates = []
    t = 0.0
    while t < max(duration - MIN_CLIP, 0.0) + step:
        start = t
        end = min(start + win, duration)
        density = sum(1 for p in peaks if start <= p < end)
        candidates.append({"start_seconds": start, "end_seconds": end, "score": density,
                           "moment_type": "climax", "label": "High-energy moment",
                           "hook": "Energy peak", "reasoning": "Densest audio energy in the piece."})
        t += step

    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Normalize density scores to a 1-100 range for consistency
    top = candidates[0]["score"] or 1
    for c in candidates:
        c["score"] = int(round(60 + 40 * (c["score"] / top)))
    return candidates


def _even_windows(duration: float) -> list[dict]:
    """Last-resort: a couple of evenly-spaced windows when no signals exist."""
    if duration <= 0:
        return []
    if duration <= MAX_CLIP:
        return [{"start_seconds": 0.0, "end_seconds": min(duration, MAX_CLIP), "score": 60,
                 "moment_type": "main_hook", "label": "Full take", "hook": "", "reasoning": "Short piece."}]
    n = min(MAX_CLIPS, max(1, int(duration // 60)))
    out = []
    for i in range(n):
        start = (duration / n) * i
        out.append({"start_seconds": start, "end_seconds": min(start + TARGET_CLIP, duration), "score": 55,
                    "moment_type": "main_hook", "label": f"Segment {i + 1}", "hook": "", "reasoning": "Evenly spaced."})
    return out


# ---------------------------------------------------------------------------
# Normalization / dedup
# ---------------------------------------------------------------------------

def _normalize(clip: dict, duration: float, onsets: list[float]) -> dict | None:
    start = _to_seconds(clip.get("start_seconds", clip.get("start_time")))
    end = _to_seconds(clip.get("end_seconds", clip.get("end_time")))
    if start is None or end is None:
        return None

    start = max(0.0, min(start, duration if duration else start))
    end = max(0.0, min(end, duration if duration else end))
    if end <= start:
        end = start + TARGET_CLIP

    # Snap to nearest onset (phrase boundary) when one is close.
    start = _snap(start, onsets)
    end = _snap(end, onsets)

    length = end - start
    if length < MIN_CLIP:
        end = start + TARGET_CLIP
    elif length > MAX_CLIP:
        end = start + MAX_CLIP
    if duration:
        end = min(end, duration)
    if end - start < MIN_CLIP:
        start = max(0.0, end - TARGET_CLIP)
    if end <= start:
        return None

    score = int(clip.get("score", 60) or 60)
    return {
        "start_time_seconds": round(start, 2),
        "end_time_seconds": round(end, 2),
        "duration_seconds": round(end - start, 2),
        "transcript_snippet": clip.get("label", ""),
        "clip_title": clip.get("label", "Musical moment"),
        "hook_text": clip.get("hook", ""),
        "virality_score": score,
        "score_breakdown": {
            "hook_strength": score,
            "standalone_clarity": score,
            "emotional_arc": score,
            "trend_alignment": max(0, score - 10),
            "rewatch_potential": score,
        },
        "reasoning": clip.get("reasoning", ""),
        "suggested_caption": "",  # post_prep regenerates in the creator's voice
        "suggested_duration": "30s" if (end - start) <= 40 else "60s",
        "clip_type": clip.get("moment_type", "main_hook"),
        "edit_suggestions": [],
    }


def _snap(t: float, onsets: list[float]) -> float:
    if not onsets:
        return t
    nearest = min(onsets, key=lambda o: abs(o - t))
    return nearest if abs(nearest - t) <= _SNAP_WINDOW else t


def _dedup(clips: list[dict]) -> list[dict]:
    """Keep higher-scored clips; drop ones that start within _OVERLAP_GAP of a kept clip."""
    kept: list[dict] = []
    for clip in sorted(clips, key=lambda c: c.get("virality_score", 0), reverse=True):
        s = clip["start_time_seconds"]
        if any(abs(s - k["start_time_seconds"]) < _OVERLAP_GAP for k in kept):
            continue
        kept.append(clip)
    kept.sort(key=lambda c: c["start_time_seconds"])
    return kept


def _to_seconds(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None
