"""Tests for the music clip-detection brain and content routing."""
from __future__ import annotations

from backend.services import music_clip_detector as mcd
from backend.services.music_clip_detector import (
    MAX_CLIP,
    MIN_CLIP,
    detect_music_clips,
    looks_instrumental,
)


# ---------------------------------------------------------------------------
# Routing: speech vs music
# ---------------------------------------------------------------------------

def test_looks_instrumental_true_for_no_speech():
    # A 3-minute violin cover with a couple of stray words
    assert looks_instrumental([{"word": "yeah"}, {"word": "ok"}], duration=180)
    assert looks_instrumental([], duration=120)


def test_looks_instrumental_false_for_talking():
    words = [{"word": "w"} for _ in range(300)]  # ~3 words/sec over 100s
    assert not looks_instrumental(words, duration=100)


def test_looks_instrumental_handles_missing_duration():
    assert looks_instrumental([{"word": "a"}] * 5, duration=0)      # few words -> music
    assert not looks_instrumental([{"word": "a"}] * 20, duration=0)  # many -> speech


# ---------------------------------------------------------------------------
# Energy fallback (no Gemini key)
# ---------------------------------------------------------------------------

def test_detect_music_clips_energy_fallback(monkeypatch):
    monkeypatch.setattr(mcd, "GEMINI_API_KEY", "")
    # Energy peaks clustered around 40s and 120s
    peaks = [{"time_seconds": t} for t in (38, 39, 40, 41, 42, 118, 120, 122)]
    signal = {"energy_peaks": peaks, "onset_times": [40.0, 120.0]}

    clips = detect_music_clips("/tmp/none.wav", signal, duration_seconds=180)

    assert clips, "energy fallback should still produce clips"
    for c in clips:
        assert MIN_CLIP <= c["duration_seconds"] <= MAX_CLIP
        assert 0 <= c["start_time_seconds"] < c["end_time_seconds"] <= 180
        assert set(c) >= {"start_time_seconds", "end_time_seconds", "clip_title", "virality_score", "clip_type"}


def test_detect_music_clips_free_mode_skips_gemini(monkeypatch):
    # Key present, but free mode -> energy fallback only, Gemini never called
    monkeypatch.setattr(mcd, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(mcd, "USE_PAID_APIS", False)

    def _boom(*a, **k):
        raise AssertionError("Gemini must not be called in free mode")

    monkeypatch.setattr(mcd, "_gemini_clips", _boom)

    signal = {"energy_peaks": [{"time_seconds": t} for t in (40, 41, 42, 120)], "onset_times": []}
    clips = detect_music_clips("/tmp/none.wav", signal, duration_seconds=180)
    assert clips


def test_detect_music_clips_short_piece(monkeypatch):
    monkeypatch.setattr(mcd, "GEMINI_API_KEY", "")
    clips = detect_music_clips("/tmp/none.wav", {"energy_peaks": [], "onset_times": []}, duration_seconds=45)
    assert len(clips) == 1
    assert clips[0]["duration_seconds"] <= MAX_CLIP


# ---------------------------------------------------------------------------
# Normalization + dedup + parsing
# ---------------------------------------------------------------------------

def test_normalize_clamps_and_enforces_duration():
    # 90s requested -> clamped to MAX_CLIP
    out = mcd._normalize({"start_seconds": 10, "end_seconds": 100, "score": 80}, duration=180, onsets=[])
    assert out["duration_seconds"] <= MAX_CLIP
    assert out["end_time_seconds"] <= 180

    # 5s requested -> extended to at least MIN_CLIP
    out2 = mcd._normalize({"start_seconds": 10, "end_seconds": 15, "score": 50}, duration=180, onsets=[])
    assert out2["duration_seconds"] >= MIN_CLIP


def test_normalize_snaps_to_onset():
    out = mcd._normalize({"start_seconds": 10.4, "end_seconds": 45, "score": 70}, duration=180, onsets=[10.0])
    assert out["start_time_seconds"] == 10.0  # snapped to the nearby onset


def test_dedup_drops_overlapping_lower_score():
    clips = [
        {"start_time_seconds": 10.0, "virality_score": 90},
        {"start_time_seconds": 14.0, "virality_score": 50},  # within 10s of the first
        {"start_time_seconds": 80.0, "virality_score": 70},
    ]
    kept = mcd._dedup(clips)
    starts = [c["start_time_seconds"] for c in kept]
    assert 10.0 in starts and 80.0 in starts and 14.0 not in starts


def test_to_seconds_parses_numbers_and_timestamps():
    assert mcd._to_seconds(90) == 90.0
    assert mcd._to_seconds("1:30") == 90.0
    assert mcd._to_seconds("bad") is None
    assert mcd._to_seconds(None) is None
