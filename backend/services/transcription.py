"""
Transcription service for ClipForge.

Uses faster-whisper (CTranslate2-based Whisper runtime) for word-level
transcription. The WhisperModel is loaded ONCE at module import time so
that multiple jobs can reuse the same in-memory model without incurring
a 3GB reload per job (see PITFALLS.md Pitfall 5).

Interface:
    transcribe_video(video_path: str, job_id: str) -> dict

Output dict:
    transcript_path  str   — path to transcript.json written in the job dir
    words            list  — [{word, start, end, probability}, ...]
    text             str   — full transcript as plain text
    duration         float — audio duration in seconds

Audio extraction:
    Raw ffmpeg subprocess call (NOT moviepy, NOT pydub).
    Output: data/downloads/{job_id}/audio.wav at 16kHz mono PCM.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from faster_whisper import WhisperModel

from backend.config import DOWNLOADS_DIR, FFMPEG_PATH, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model — loaded ONCE at import time.
# PITFALLS.md Pitfall 5: loading inside transcribe_video() causes 3GB reloads.
# ---------------------------------------------------------------------------
_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", settings.whisper_model)

logger.info(f"Loading Whisper model: {_WHISPER_MODEL_NAME}")
_model = WhisperModel(_WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
logger.info(f"Whisper model ready: {_WHISPER_MODEL_NAME}")


def transcribe_video(video_path: str, job_id: str) -> dict:
    """Transcribe a video file, producing word-level timestamps.

    Args:
        video_path: Absolute path to the video file (e.g. .mp4).
        job_id: Unique job identifier — used to determine output directory.

    Returns:
        dict with keys:
            transcript_path (str): Path to transcript.json on disk.
            words (list[dict]): Each dict has {word, start, end, probability}.
            text (str): Full transcript as plain text.
            duration (float): Audio duration in seconds.

    Side effects:
        Writes:
            data/downloads/{job_id}/audio.wav  — extracted 16kHz mono audio
            data/downloads/{job_id}/transcript.json  — word-level transcript
    """
    job_dir = DOWNLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = str(job_dir / "audio.wav")
    transcript_path = job_dir / "transcript.json"

    # --- Step 1: Extract audio using raw ffmpeg subprocess ---
    # -ar 16000: resample to 16kHz (Whisper requirement)
    # -ac 1:     mono channel
    # -c:a pcm_s16le: uncompressed 16-bit PCM — fastest for Whisper
    ffmpeg_cmd = [
        FFMPEG_PATH,
        "-y",             # overwrite if exists
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        audio_path,
    ]

    logger.info(f"[{job_id}] Extracting audio: {video_path} -> {audio_path}")
    result = subprocess.run(
        ffmpeg_cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        logger.warning(f"[{job_id}] ffmpeg audio extraction failed (rc={result.returncode}): {result.stderr[:300]}")
        # Fall back to passing video directly to Whisper (it can decode many formats)
        audio_input = video_path
    else:
        audio_input = audio_path

    # --- Step 2: Transcribe with faster-whisper ---
    # _model is loaded at module level — NOT called here (PITFALLS.md Pitfall 5)
    logger.info(f"[{job_id}] Transcribing: {audio_input}")

    segments_gen, info = _model.transcribe(
        audio_input,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    # --- Step 3: Flatten words from all segments ---
    words: list[dict] = []
    full_text_parts: list[str] = []

    for segment in segments_gen:
        seg_text = segment.text.strip()
        if seg_text:
            full_text_parts.append(seg_text)

        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "probability": round(word.probability, 3),
                })

    full_text = " ".join(full_text_parts)
    duration = round(info.duration, 2)

    logger.info(
        f"[{job_id}] Transcription complete: {len(words)} words, "
        f"{info.language} ({round(info.language_probability, 2)}), "
        f"{duration}s"
    )

    # --- Step 4: Write transcript.json to job directory ---
    transcript_data = {
        "words": words,
        "text": full_text,
        "duration": duration,
    }

    transcript_path.write_text(
        json.dumps(transcript_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"[{job_id}] Transcript written: {transcript_path}")

    return {
        "transcript_path": str(transcript_path),
        "words": words,
        "text": full_text,
        "duration": duration,
    }
