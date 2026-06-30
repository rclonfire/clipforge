"""
Clip detection service for ClipForge — Phase 02-02 rebuild.

Detects the best short-form clip candidates from a video using Claude.
Handles long videos via chunked transcript analysis with overlap and deduplication.

Interface:
    detect_clips(words, signal_data, duration_seconds) -> list[dict]

Key behaviors:
- Videos <= 10 min: single Claude call with full transcript
- Videos > 10 min: overlapping 10-min chunks with 2-min overlap
- Clips within 5 seconds of each other: keep only the higher-scored one
- Output constrained to 5-8 clips, 30-60 second duration
  (relaxes to 20-90s if fewer than 5 survive strict range)
"""
from __future__ import annotations

import json
import logging

import anthropic

from backend.config import ANTHROPIC_API_KEY
from backend.prompts.clip_detection_system import CLIP_DETECTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "claude-sonnet-4-6"

CHUNK_DURATION_SECONDS = 600   # 10 minutes per chunk
OVERLAP_SECONDS = 120          # 2-minute overlap between chunks
MIN_CLIP_DURATION = 30
MAX_CLIP_DURATION = 60
MIN_CLIPS = 5
MAX_CLIPS = 8

# Relaxed duration range when strict filter produces too few clips
_RELAXED_MIN = 20
_RELAXED_MAX = 90

# Group words into ~5-second segments for compact transcript formatting
_SEGMENT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def detect_clips(
    words: list[dict],
    signal_data: dict,
    duration_seconds: float,
) -> list[dict]:
    """
    Use Claude to detect the best short-form clip candidates from a video.

    Args:
        words: Word-level transcript dicts [{word, start, end, probability}, ...]
        signal_data: Output from analyze_signals() — energy peaks, silence gaps, etc.
        duration_seconds: Total video duration (used to decide whether to chunk).

    Returns:
        List of clip dicts, each containing start_time_seconds, end_time_seconds,
        duration_seconds, transcript_snippet, clip_title, hook_text, virality_score,
        score_breakdown, reasoning, suggested_caption, suggested_duration,
        clip_type, edit_suggestions.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    chunks = _build_chunks(words, duration_seconds)
    total_chunks = len(chunks)

    all_clips: list[dict] = []

    for i, chunk_words in enumerate(chunks, start=1):
        chunk_start = chunk_words[0]["start"] if chunk_words else 0.0
        chunk_end = chunk_words[-1]["end"] if chunk_words else duration_seconds

        logger.info("Processing chunk %d/%d", i, total_chunks)

        formatted_transcript = _format_transcript_for_claude(chunk_words)

        chunk_start_mm = int(chunk_start // 60)
        chunk_start_ss = int(chunk_start % 60)
        chunk_end_mm = int(chunk_end // 60)
        chunk_end_ss = int(chunk_end % 60)

        user_message = (
            f"TIMESTAMPED TRANSCRIPT (chunk {i} of {total_chunks}, "
            f"{chunk_start_mm}:{chunk_start_ss:02d} - {chunk_end_mm}:{chunk_end_ss:02d}):\n"
            f"{formatted_transcript}\n\n"
            f"AUDIO SIGNAL DATA:\n"
            f"{json.dumps(signal_data, indent=2)}"
        )

        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8000,
                system=CLIP_DETECTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            response_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```) and last line (```)
                response_text = "\n".join(lines[1:-1]).strip()

            result = json.loads(response_text)
            raw_clips = result.get("clips", []) if isinstance(result, dict) else result

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Could not parse Claude response for chunk %d: %s", i, exc)
            raw_clips = []

        # Parse time strings to seconds and compute duration
        chunk_clips = []
        for clip in raw_clips:
            clip["start_time_seconds"] = _time_to_seconds(clip.get("start_time", "0:00"))
            clip["end_time_seconds"] = _time_to_seconds(clip.get("end_time", "0:00"))
            if not clip.get("duration_seconds"):
                clip["duration_seconds"] = (
                    clip["end_time_seconds"] - clip["start_time_seconds"]
                )
            chunk_clips.append(clip)

        logger.info("Detected %d clips from chunk %d/%d", len(chunk_clips), i, total_chunks)
        all_clips.extend(chunk_clips)

    deduped = _dedup_clips(all_clips)
    logger.info("After dedup: %d clips", len(deduped))

    final_clips = _enforce_constraints(deduped)
    logger.info("Final: %d clips returned", len(final_clips))

    return final_clips


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_chunks(words: list[dict], duration_seconds: float) -> list[list[dict]]:
    """
    Return [words] for short videos; overlapping 10-min chunks for long ones.

    For videos <= CHUNK_DURATION_SECONDS: returns a single-element list.
    For longer videos: advances start by (CHUNK_DURATION_SECONDS - OVERLAP_SECONDS)
    each iteration, includes all words in [chunk_start, chunk_start + CHUNK_DURATION_SECONDS).
    """
    if duration_seconds <= CHUNK_DURATION_SECONDS:
        return [words]

    chunks: list[list[dict]] = []
    advance = CHUNK_DURATION_SECONDS - OVERLAP_SECONDS  # 480 seconds
    chunk_start = 0.0

    while chunk_start < duration_seconds:
        chunk_end = chunk_start + CHUNK_DURATION_SECONDS
        chunk = [w for w in words if w["start"] >= chunk_start and w["start"] < chunk_end]
        if chunk:
            chunks.append(chunk)
        chunk_start += advance

    return chunks if chunks else [words]


def _format_transcript_for_claude(chunk_words: list[dict]) -> str:
    """
    Group words into ~5-second segments and format as '[M:SS] word word word'.

    Strips probability values — only uses word, start, end.
    Returns a newline-joined string of segment lines.
    """
    if not chunk_words:
        return ""

    lines: list[str] = []
    segment_words: list[str] = []
    segment_start: float = chunk_words[0]["start"]

    for word_dict in chunk_words:
        t = word_dict["start"]
        text = word_dict["word"]

        # Start a new segment when we've exceeded the segment window
        if t - segment_start >= _SEGMENT_SECONDS and segment_words:
            minutes = int(segment_start // 60)
            seconds = int(segment_start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {' '.join(segment_words)}")
            segment_words = []
            segment_start = t

        segment_words.append(text)

    # Flush the last segment
    if segment_words:
        minutes = int(segment_start // 60)
        seconds = int(segment_start % 60)
        lines.append(f"[{minutes}:{seconds:02d}] {' '.join(segment_words)}")

    return "\n".join(lines)


def _dedup_clips(all_clips: list[dict]) -> list[dict]:
    """
    Remove clips whose start_time_seconds is within 5 seconds of a higher-scored clip.

    Sort by virality_score descending, then greedily keep clips that don't
    fall within 5 seconds of an already-kept clip.
    """
    sorted_clips = sorted(
        all_clips,
        key=lambda c: c.get("virality_score", 0),
        reverse=True,
    )

    kept: list[dict] = []
    for clip in sorted_clips:
        clip_start = clip.get("start_time_seconds", 0.0)
        too_close = any(
            abs(clip_start - kept_clip.get("start_time_seconds", 0.0)) < 5.0
            for kept_clip in kept
        )
        if not too_close:
            kept.append(clip)

    return kept


def _enforce_constraints(clips: list[dict]) -> list[dict]:
    """
    Filter clips to the 30-60s duration range, cap at MAX_CLIPS.

    If fewer than MIN_CLIPS survive the strict filter, relax to 20-90s range.
    Cap is always applied at MAX_CLIPS regardless of which range is used.
    """
    # Strict filter: 30-60 seconds
    strict = [
        c for c in clips
        if MIN_CLIP_DURATION <= c.get("duration_seconds", 0) <= MAX_CLIP_DURATION
    ]

    if len(strict) >= MIN_CLIPS:
        return strict[:MAX_CLIPS]

    # Relax to 20-90 seconds
    relaxed = [
        c for c in clips
        if _RELAXED_MIN <= c.get("duration_seconds", 0) <= _RELAXED_MAX
    ]
    return relaxed[:MAX_CLIPS]


def _time_to_seconds(time_str: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' to float seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0
