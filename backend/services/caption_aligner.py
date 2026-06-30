"""
Caption aligner service for ClipForge — Phase 03-export-engine.

Runs WhisperX forced alignment on an audio file to produce word-level
timestamps that are more precise than faster-whisper's native word timestamps.
Falls back to the original faster-whisper word timestamps when WhisperX
alignment returns empty results.

Interface:
    align_words_for_clip(audio_path, words, clip_start, clip_end, language, device)
        -> list[dict]  — [{word, start, end}, ...] with timestamps relative to clip start

Key behaviors:
- Model is cached per (language, device) pair to avoid reloading on every clip
- Words are grouped into utterance-level segments for whisperx.align() input
- Words outside clip window [clip_start, clip_end] are filtered out
- Words missing 'start' key (unaligned by WhisperX) are filtered out
- On empty alignment result, falls back to faster-whisper's word timestamps
- All returned timestamps are clip-relative (offset by subtracting clip_start)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level alignment model cache — keyed by (language, device)
_align_model_cache: dict[tuple[str, str], tuple[Any, Any]] = {}


def get_align_model(language: str = "en", device: str = "cpu") -> tuple[Any, Any]:
    """
    Load and cache the WhisperX alignment model for the given language/device.

    Returns:
        (model_a, metadata) tuple — passed directly to whisperx.align()
    """
    import whisperx

    cache_key = (language, device)
    if cache_key not in _align_model_cache:
        logger.info("Loading WhisperX alignment model: lang=%s device=%s", language, device)
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        _align_model_cache[cache_key] = (model_a, metadata)
    return _align_model_cache[cache_key]


def _words_to_segments(
    words: list[dict],
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """
    Reconstruct utterance-level segments from word dicts for whisperx.align() input format.

    Groups consecutive words into segments (starts a new segment when the gap
    between words is > 1.0 seconds). Filters to clip window [clip_start, clip_end].

    Returns:
        list of {"start": float, "end": float, "text": str} dicts
    """
    # Filter to clip window first
    clip_words = [
        w for w in words
        if "start" in w and "end" in w
        and clip_start <= w["start"] <= clip_end
    ]

    if not clip_words:
        return []

    segments: list[dict] = []
    current_segment_words: list[dict] = []
    prev_end: float | None = None

    for w in clip_words:
        if prev_end is not None and (w["start"] - prev_end) > 1.0:
            # Gap > 1.0s — close current segment and start a new one
            if current_segment_words:
                segments.append({
                    "start": current_segment_words[0]["start"],
                    "end": current_segment_words[-1]["end"],
                    "text": " ".join(cw["word"] for cw in current_segment_words),
                })
            current_segment_words = [w]
        else:
            current_segment_words.append(w)
        prev_end = w["end"]

    # Flush final segment
    if current_segment_words:
        segments.append({
            "start": current_segment_words[0]["start"],
            "end": current_segment_words[-1]["end"],
            "text": " ".join(cw["word"] for cw in current_segment_words),
        })

    return segments


def _fallback_to_faster_whisper_words(
    words: list[dict],
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """
    Filter original faster-whisper words list to clip window and offset timestamps
    to be clip-relative (subtract clip_start).

    Args:
        words: List of word dicts with keys {word, start, end, probability}.
        clip_start: Clip start time in seconds (absolute).
        clip_end: Clip end time in seconds (absolute).

    Returns:
        list of {word, start, end} dicts with clip-relative timestamps.
    """
    result: list[dict] = []
    for w in words:
        if "start" not in w or "end" not in w:
            continue
        if not (clip_start <= w["start"] <= clip_end):
            continue
        result.append({
            "word": w["word"],
            "start": round(w["start"] - clip_start, 3),
            "end": round(w["end"] - clip_start, 3),
        })
    return result


def align_words_for_clip(
    audio_path: str,
    words: list[dict],
    clip_start: float,
    clip_end: float,
    language: str = "en",
    device: str = "cpu",
) -> list[dict]:
    """
    Run WhisperX forced alignment and return word-level timestamps for a clip.

    All returned timestamps are relative to clip_start (i.e., clip_start is
    subtracted from each word's start/end).

    If WhisperX alignment returns no words in the clip window, falls back to
    the original faster-whisper word timestamps from `words`.

    Args:
        audio_path: Path to the audio file (wav/mp3) covering at least the clip window.
        words: Original faster-whisper word list (fallback source) with absolute timestamps.
        clip_start: Clip start time in seconds (absolute).
        clip_end: Clip end time in seconds (absolute).
        language: Language code for WhisperX alignment model (default "en").
        device: Compute device for alignment model ("cpu" or "cuda").

    Returns:
        list of {word: str, start: float, end: float} dicts, timestamps relative to clip_start.
    """
    import whisperx

    # Load audio and alignment model
    audio = whisperx.load_audio(audio_path)
    model_a, metadata = get_align_model(language, device)

    # Build utterance-level segments from the word list for WhisperX
    segments = _words_to_segments(words, clip_start, clip_end)

    # Run WhisperX forced alignment
    alignment_result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Extract and filter word segments
    aligned_words: list[dict] = []
    for w in alignment_result.get("word_segments", []):
        if "start" not in w or "end" not in w:
            continue  # Unaligned word
        if not (clip_start <= w["start"] <= clip_end):
            continue  # Outside clip window
        aligned_words.append({
            "word": w["word"],
            "start": round(w["start"] - clip_start, 3),
            "end": round(w["end"] - clip_start, 3),
        })

    # Fallback if alignment returned nothing in the clip window
    if not aligned_words:
        logger.warning(
            "WhisperX returned empty alignment for clip [%.2f, %.2f]; falling back to faster-whisper words.",
            clip_start, clip_end,
        )
        return _fallback_to_faster_whisper_words(words, clip_start, clip_end)

    return aligned_words
