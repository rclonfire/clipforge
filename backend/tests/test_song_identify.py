"""Tests for autonomous song identification and its gating."""
from __future__ import annotations

from backend.services import song_identify
from backend.services.song_identify import is_named


# ---------------------------------------------------------------------------
# is_named gate
# ---------------------------------------------------------------------------

def test_is_named_true_for_high_and_medium_with_song():
    assert is_named({"song": "Snooze", "artist": "SZA", "confidence": "high"})
    assert is_named({"song": "Iris", "artist": "Goo Goo Dolls", "confidence": "medium"})


def test_is_named_false_for_low_none_or_missing_song():
    assert not is_named({"song": "Snooze", "confidence": "low"})
    assert not is_named({"song": "Snooze", "confidence": "none"})
    assert not is_named({"song": "", "confidence": "high"})  # no title → not named
    assert not is_named(None)
    assert not is_named({})


# ---------------------------------------------------------------------------
# identify_song degradation (no network / no key / missing file)
# ---------------------------------------------------------------------------

def test_identify_song_without_key_returns_none(monkeypatch):
    monkeypatch.setattr(song_identify, "GEMINI_API_KEY", "")
    result = song_identify.identify_song("/any/path.wav")
    assert result["confidence"] == "none"
    assert result["song"] == "" and result["artist"] == ""
    assert not is_named(result)


def test_identify_song_missing_file_returns_none(monkeypatch):
    monkeypatch.setattr(song_identify, "GEMINI_API_KEY", "test-key")
    result = song_identify.identify_song("/no/such/audio.wav")
    assert result["confidence"] == "none"
    assert not is_named(result)


def test_identify_song_recovers_from_gemini_error(monkeypatch, tmp_path):
    # Audio exists; segment extraction + Gemini both raise → graceful 'none'
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not really audio")
    monkeypatch.setattr(song_identify, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(song_identify, "_extract_segment", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg")))
    monkeypatch.setattr(song_identify, "_ask_gemini", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")))
    result = song_identify.identify_song(str(audio))
    assert result["confidence"] == "none"
    assert not is_named(result)
