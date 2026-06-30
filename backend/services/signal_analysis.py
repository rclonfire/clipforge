"""
Signal analysis service for ClipForge — Phase 02 implementation.

Uses librosa for frame-level audio analysis (~10ms resolution) instead of
FFmpeg astats (1-second buckets). This resolution is required for comedy
timing detection: laugh bursts, punchline energy spikes, and clap onsets
occur at sub-second granularity that astats cannot capture.

Interface:
    analyze_signals(audio_path: str, words: list[dict]) -> dict

Output dict:
    energy_peaks  list  — [{"time_seconds": float, "type": "energy_spike"}, ...]  (capped at 50)
    silence_gaps  list  — [{"start": float, "end": float, "duration": float}, ...]
    speech_rate   list  — [{"time": "MM:SS", "wpm": int, "label": str}, ...]
    onset_times   list  — [float, ...]  (capped at 100)

Audio input:
    audio_path — path to audio.wav already on disk from Phase 1 transcription.
    words      — list of word dicts from transcription: [{word, start, end, probability}, ...]

No FFmpeg subprocess calls. No scene_changes output.
"""
from __future__ import annotations

import logging
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Output caps (keeps Claude's token budget under control)
_MAX_ENERGY_PEAKS = 50
_MAX_ONSET_TIMES = 100

# Silence gap threshold in seconds
_SILENCE_GAP_THRESHOLD = 1.5

# Speech rate window size in seconds
_SPEECH_RATE_WINDOW = 10.0

# Hop length for librosa analysis (512 samples @ 22050 Hz ≈ 23ms per frame)
_HOP_LENGTH = 512


def analyze_signals(audio_path: str, words: list[dict]) -> dict:
    """
    Analyze audio signals from audio_path for clip detection.

    Args:
        audio_path: Path to a WAV file (reuses audio.wav from Phase 1 transcription).
        words: Word-level transcript dicts [{word, start, end, probability}, ...].

    Returns:
        {
            "energy_peaks": [{"time_seconds": float, "type": "energy_spike"}, ...],  # capped at 50
            "silence_gaps": [{"start": float, "end": float, "duration": float}, ...],
            "speech_rate": [{"time": "MM:SS", "wpm": int, "label": str}, ...],
            "onset_times": [float, ...],  # capped at 100
        }
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    energy_peaks = _detect_energy_peaks(y=y, sr=sr)
    onset_times = _detect_onset_times(y=y, sr=sr)
    silence_gaps = _detect_silence_gaps_from_words(words=words)
    speech_rate = _compute_speech_rate(words=words)

    logger.info(
        "Signal analysis complete: %d energy peaks, %d onset times, %d silence gaps",
        len(energy_peaks),
        len(onset_times),
        len(silence_gaps),
    )

    return {
        "energy_peaks": energy_peaks,
        "silence_gaps": silence_gaps,
        "speech_rate": speech_rate,
        "onset_times": onset_times,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_energy_peaks(y: np.ndarray, sr: int) -> list[dict]:
    """
    Detect audio energy peaks using librosa RMS envelope.

    Uses 80th-percentile threshold to find frames with above-average energy.
    Results are capped at _MAX_ENERGY_PEAKS to control Claude token usage.
    """
    rms = librosa.feature.rms(y=y, hop_length=_HOP_LENGTH)[0]

    if rms.size == 0:
        return []

    threshold = float(np.percentile(rms, 80))

    # Find frame indices above threshold
    peak_frames = np.where(rms >= threshold)[0]

    # Convert frames to times
    times = librosa.frames_to_time(peak_frames, sr=sr, hop_length=_HOP_LENGTH)

    peaks = [
        {"time_seconds": float(t), "type": "energy_spike"}
        for t in times
    ]

    return peaks[:_MAX_ENERGY_PEAKS]


def _detect_onset_times(y: np.ndarray, sr: int) -> list[float]:
    """
    Detect sudden audio events (laugh bursts, claps) using librosa onset detection.

    Results are capped at _MAX_ONSET_TIMES to control Claude token usage.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=_HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP_LENGTH).tolist()
    return onset_times[:_MAX_ONSET_TIMES]


def _detect_silence_gaps_from_words(words: list[dict]) -> list[dict]:
    """
    Detect silence gaps > 1.5 seconds between consecutive words.

    Uses the words list directly (word["start"] and word["end"]) rather than
    transcript segments — words have finer granularity than segments.
    """
    if len(words) < 2:
        return []

    gaps = []
    for i in range(1, len(words)):
        gap_start = words[i - 1]["end"]
        gap_end = words[i]["start"]
        gap_duration = gap_end - gap_start

        if gap_duration > _SILENCE_GAP_THRESHOLD:
            gaps.append({
                "start": float(gap_start),
                "end": float(gap_end),
                "duration": round(float(gap_duration), 2),
            })

    return gaps


def _compute_speech_rate(words: list[dict]) -> list[dict]:
    """
    Calculate words-per-minute in 10-second sliding windows.

    Label thresholds:
        fast   > 180 wpm
        slow   < 80 wpm
        normal otherwise
    """
    if not words:
        return []

    max_time = words[-1]["end"]
    results = []

    t = 0.0
    while t < max_time:
        window_end = t + _SPEECH_RATE_WINDOW
        words_in_window = [
            w for w in words if w["start"] >= t and w["start"] < window_end
        ]
        # wpm: words per 10 seconds * 6 = words per minute
        wpm = int(round(len(words_in_window) * (60.0 / _SPEECH_RATE_WINDOW)))

        if wpm > 180:
            label = "fast"
        elif wpm < 80:
            label = "slow"
        else:
            label = "normal"

        minutes = int(t // 60)
        seconds = int(t % 60)
        results.append({
            "time": f"{minutes:02d}:{seconds:02d}",
            "wpm": wpm,
            "label": label,
        })
        t += _SPEECH_RATE_WINDOW

    return results
