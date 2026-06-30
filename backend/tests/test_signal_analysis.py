"""
Tests for Phase 02 Plan 01: signal_analysis.py rebuilt with librosa.
TDD RED phase — defines expected behavior before implementation.

Tests use synthetic audio (numpy sine wave) — no real audio file needed.
librosa is NOT mocked; tests call the real function with real audio arrays.
"""
import io
import math
import tempfile
import os
import numpy as np
import soundfile as sf
import pytest


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def _make_sine_wav(duration: float = 10.0, sr: int = 22050, freq: float = 440.0) -> str:
    """Generate a sine wave WAV file and return its path (tmp file, caller must delete)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * math.pi * freq * t).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    tmp.close()
    return tmp.name


def _make_words(n: int = 20, total_duration: float = 20.0) -> list[dict]:
    """Generate n evenly-spaced word dicts over total_duration seconds."""
    spacing = total_duration / n
    words = []
    for i in range(n):
        start = i * spacing
        end = start + spacing * 0.5  # word lasts half the spacing
        words.append({
            "word": f"word{i}",
            "start": float(start),
            "end": float(end),
            "probability": 0.95,
        })
    return words


# ---------------------------------------------------------------------------
# Test 1: Required keys
# ---------------------------------------------------------------------------

class TestAnalyzeSignalsRequiredKeys:
    """analyze_signals must return a dict with all four required keys."""

    def test_analyze_signals_returns_required_keys(self):
        """Return dict must have: energy_peaks, silence_gaps, speech_rate, onset_times."""
        from backend.services.signal_analysis import analyze_signals

        audio_path = _make_sine_wav(duration=5.0)
        try:
            words = _make_words(n=10, total_duration=5.0)
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "energy_peaks" in result, f"Missing 'energy_peaks'. Keys: {list(result.keys())}"
        assert "silence_gaps" in result, f"Missing 'silence_gaps'. Keys: {list(result.keys())}"
        assert "speech_rate" in result, f"Missing 'speech_rate'. Keys: {list(result.keys())}"
        assert "onset_times" in result, f"Missing 'onset_times'. Keys: {list(result.keys())}"


# ---------------------------------------------------------------------------
# Test 2: Energy peaks shape
# ---------------------------------------------------------------------------

class TestEnergyPeaksShape:
    """Each energy_peak must be a dict with 'time_seconds' (float) and 'type' == 'energy_spike'."""

    def test_energy_peaks_are_dicts_with_time_seconds(self):
        """energy_peaks items must have 'time_seconds' (float) and type == 'energy_spike'."""
        from backend.services.signal_analysis import analyze_signals

        audio_path = _make_sine_wav(duration=5.0)
        try:
            words = _make_words(n=5, total_duration=5.0)
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        peaks = result["energy_peaks"]
        assert isinstance(peaks, list), f"energy_peaks must be a list, got {type(peaks)}"

        for peak in peaks:
            assert isinstance(peak, dict), f"Each peak must be a dict, got {type(peak)}"
            assert "time_seconds" in peak, f"Peak missing 'time_seconds': {peak}"
            assert isinstance(peak["time_seconds"], float), (
                f"time_seconds must be float, got {type(peak['time_seconds'])}"
            )
            assert "type" in peak, f"Peak missing 'type': {peak}"
            assert peak["type"] == "energy_spike", (
                f"Peak type must be 'energy_spike', got {peak['type']!r}"
            )


# ---------------------------------------------------------------------------
# Test 3: Onset times are floats
# ---------------------------------------------------------------------------

class TestOnsetTimesAreFloats:
    """Every item in onset_times must be a float."""

    def test_onset_times_are_floats(self):
        """onset_times must be a list of floats."""
        from backend.services.signal_analysis import analyze_signals

        audio_path = _make_sine_wav(duration=5.0)
        try:
            words = _make_words(n=5, total_duration=5.0)
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        onset_times = result["onset_times"]
        assert isinstance(onset_times, list), (
            f"onset_times must be a list, got {type(onset_times)}"
        )
        for t in onset_times:
            assert isinstance(t, float), (
                f"Each onset_time must be a float, got {type(t)} for value {t!r}"
            )


# ---------------------------------------------------------------------------
# Test 4: Silence gaps detected
# ---------------------------------------------------------------------------

class TestSilenceGapsDetected:
    """Provide words with a 2-second gap; at least one silence_gap must have duration >= 1.5."""

    def test_silence_gaps_detected(self):
        """A 2-second gap between words must produce at least one silence_gap with duration >= 1.5."""
        from backend.services.signal_analysis import analyze_signals

        audio_path = _make_sine_wav(duration=10.0)
        try:
            # Build words with explicit 2-second gap at second 5
            words = [
                {"word": "hello", "start": 3.0, "end": 3.5, "probability": 0.95},
                {"word": "there", "start": 4.0, "end": 4.5, "probability": 0.95},
                # Gap: 4.5 -> 6.5 (2 seconds)
                {"word": "world", "start": 6.5, "end": 7.0, "probability": 0.95},
                {"word": "again", "start": 7.5, "end": 8.0, "probability": 0.95},
            ]
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        gaps = result["silence_gaps"]
        assert isinstance(gaps, list), f"silence_gaps must be a list, got {type(gaps)}"
        assert len(gaps) >= 1, (
            f"Expected at least 1 silence gap (2-second gap was provided). "
            f"Got 0 gaps. silence_gaps: {gaps}"
        )
        durations = [g["duration"] for g in gaps]
        assert any(d >= 1.5 for d in durations), (
            f"Expected at least one gap with duration >= 1.5. Got durations: {durations}"
        )


# ---------------------------------------------------------------------------
# Test 5: Speech rate windows
# ---------------------------------------------------------------------------

class TestSpeechRateWindows:
    """speech_rate must be non-empty with each item having 'time', 'wpm', 'label'."""

    def test_speech_rate_windows(self):
        """20 words over 20 seconds must produce speech_rate list with correct item shape."""
        from backend.services.signal_analysis import analyze_signals

        audio_path = _make_sine_wav(duration=20.0)
        try:
            words = _make_words(n=20, total_duration=20.0)
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        sr_list = result["speech_rate"]
        assert isinstance(sr_list, list), f"speech_rate must be a list, got {type(sr_list)}"
        assert len(sr_list) > 0, "speech_rate must be non-empty for 20 words over 20 seconds"

        for item in sr_list:
            assert isinstance(item, dict), f"Each speech_rate item must be a dict, got {type(item)}"
            assert "time" in item, f"Missing 'time' key in speech_rate item: {item}"
            assert "wpm" in item, f"Missing 'wpm' key in speech_rate item: {item}"
            assert "label" in item, f"Missing 'label' key in speech_rate item: {item}"
            assert isinstance(item["time"], str), (
                f"'time' must be a string (MM:SS format), got {type(item['time'])}"
            )
            assert isinstance(item["wpm"], int), (
                f"'wpm' must be an int, got {type(item['wpm'])}"
            )
            assert item["label"] in ("fast", "normal", "slow"), (
                f"'label' must be 'fast', 'normal', or 'slow'. Got: {item['label']!r}"
            )


# ---------------------------------------------------------------------------
# Test 6: Output caps enforced
# ---------------------------------------------------------------------------

class TestOutputCaps:
    """energy_peaks must be capped at 50; onset_times must be capped at 100."""

    def test_output_caps(self):
        """energy_peaks len <= 50 and onset_times len <= 100 regardless of audio content."""
        from backend.services.signal_analysis import analyze_signals

        # Generate 30 seconds of loud sine wave — should produce many peaks
        duration = 30.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Mix several frequencies to maximize onset detection
        y = (
            0.8 * np.sin(2 * math.pi * 440.0 * t)
            + 0.2 * np.sin(2 * math.pi * 880.0 * t)
        ).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            audio_path = f.name

        try:
            words = _make_words(n=30, total_duration=30.0)
            result = analyze_signals(audio_path=audio_path, words=words)
        finally:
            os.unlink(audio_path)

        peaks = result["energy_peaks"]
        onset_times = result["onset_times"]

        assert len(peaks) <= 50, (
            f"energy_peaks must be capped at 50. Got {len(peaks)} peaks."
        )
        assert len(onset_times) <= 100, (
            f"onset_times must be capped at 100. Got {len(onset_times)} onset times."
        )
