"""
Tests for Plan 02-02: Rebuilt clip_detector.py with chunking, dedup, and constraints.
TDD RED phase — defines expected behavior before implementation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: Model ID
# ---------------------------------------------------------------------------

def test_model_id():
    """detect_clips must call Claude with model="claude-sonnet-4-6"."""
    from backend.services.clip_detector import detect_clips

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({"clips": []})

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    words = [
        {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.98},
    ]
    signal_data = {"energy_peaks": [], "silence_gaps": [], "speech_rate": [], "onset_times": []}

    with patch("backend.services.clip_detector.anthropic.Anthropic", return_value=mock_client):
        detect_clips(words=words, signal_data=signal_data, duration_seconds=60.0)

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6", (
        f"Expected model='claude-sonnet-4-6', got model='{call_kwargs.get('model')}'"
    )


# ---------------------------------------------------------------------------
# Test 2: _build_chunks short video (no chunking needed)
# ---------------------------------------------------------------------------

def test_build_chunks_short_video():
    """_build_chunks must return a single chunk for videos <= 600 seconds."""
    from backend.services.clip_detector import _build_chunks

    # 300 words over 300 seconds (5 min)
    words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.5, "probability": 0.99}
        for i in range(300)
    ]
    duration_seconds = 300.0

    chunks = _build_chunks(words, duration_seconds)

    assert isinstance(chunks, list), "_build_chunks must return a list"
    assert len(chunks) == 1, (
        f"Short video (300s) must produce exactly 1 chunk, got {len(chunks)}"
    )
    assert chunks[0] == words, "Single chunk must contain all words"


# ---------------------------------------------------------------------------
# Test 3: _build_chunks long video (overlapping chunks)
# ---------------------------------------------------------------------------

def test_build_chunks_long_video():
    """_build_chunks must return 3+ overlapping chunks for 20-minute videos."""
    from backend.services.clip_detector import _build_chunks

    # Words covering 1200 seconds (20 min), one word per second
    words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.5, "probability": 0.99}
        for i in range(1200)
    ]
    duration_seconds = 1200.0

    chunks = _build_chunks(words, duration_seconds)

    assert isinstance(chunks, list), "_build_chunks must return a list"
    assert len(chunks) >= 3, (
        f"20-min video must produce 3+ chunks, got {len(chunks)}"
    )

    # Check overlap: first word of chunk N+1 must appear in chunk N
    for idx in range(len(chunks) - 1):
        chunk_n = chunks[idx]
        chunk_next = chunks[idx + 1]
        if not chunk_next:
            continue
        first_word_next = chunk_next[0]
        # The first word of the next chunk should appear in the current chunk (overlap)
        words_in_chunk_n = {w["start"] for w in chunk_n}
        assert first_word_next["start"] in words_in_chunk_n, (
            f"Chunk {idx} and chunk {idx + 1} must overlap: "
            f"first word of chunk {idx + 1} (start={first_word_next['start']}) "
            f"not found in chunk {idx}"
        )


# ---------------------------------------------------------------------------
# Test 4: _dedup_clips removes nearby lower-scored clips
# ---------------------------------------------------------------------------

def test_dedup_removes_nearby_clips():
    """_dedup_clips must keep the higher-scored clip when two are within 5 seconds."""
    from backend.services.clip_detector import _dedup_clips

    clip_high = {
        "start_time_seconds": 100.0,
        "end_time_seconds": 145.0,
        "duration_seconds": 45.0,
        "virality_score": 80,
        "clip_title": "High score clip",
    }
    clip_low = {
        "start_time_seconds": 102.0,  # 2 seconds from high-score clip (within 5s)
        "end_time_seconds": 147.0,
        "duration_seconds": 45.0,
        "virality_score": 60,
        "clip_title": "Low score clip",
    }

    result = _dedup_clips([clip_high, clip_low])

    assert len(result) == 1, f"Expected 1 clip after dedup, got {len(result)}"
    assert result[0]["virality_score"] == 80, (
        f"Expected score=80 to survive, got score={result[0]['virality_score']}"
    )


# ---------------------------------------------------------------------------
# Test 5: _enforce_constraints caps at 8 clips
# ---------------------------------------------------------------------------

def test_enforce_constraints_caps_at_eight():
    """_enforce_constraints must cap the output at 8 clips."""
    from backend.services.clip_detector import _enforce_constraints

    # 12 clips all with 45s duration (within 30-60s range)
    clips = [
        {
            "start_time_seconds": float(i * 60),
            "end_time_seconds": float(i * 60 + 45),
            "duration_seconds": 45.0,
            "virality_score": 50 + i,
        }
        for i in range(12)
    ]

    result = _enforce_constraints(clips)

    assert len(result) == 8, f"Expected 8 clips after cap, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 6: _enforce_constraints relaxes duration when fewer than MIN_CLIPS survive
# ---------------------------------------------------------------------------

def test_enforce_constraints_relaxes_duration():
    """_enforce_constraints must relax to 20-90s range when fewer than 5 clips survive 30-60s filter."""
    from backend.services.clip_detector import _enforce_constraints

    # 3 clips with 25s duration (below 30s minimum) — too few for strict range
    clips = [
        {
            "start_time_seconds": float(i * 60),
            "end_time_seconds": float(i * 60 + 25),
            "duration_seconds": 25.0,
            "virality_score": 70,
        }
        for i in range(3)
    ]

    result = _enforce_constraints(clips)

    assert len(result) == 3, (
        f"All 3 clips should survive when duration constraint is relaxed, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Test 7: _format_transcript_for_claude strips probability
# ---------------------------------------------------------------------------

def test_format_transcript_strips_probability():
    """_format_transcript_for_claude must not include 'probability' in output."""
    from backend.services.clip_detector import _format_transcript_for_claude

    words = [
        {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.98},
        {"word": "this", "start": 1.1, "end": 1.5, "probability": 0.97},
        {"word": "is", "start": 1.6, "end": 1.8, "probability": 0.99},
        {"word": "a", "start": 1.9, "end": 2.0, "probability": 0.99},
        {"word": "test", "start": 2.1, "end": 2.5, "probability": 0.95},
    ]

    output = _format_transcript_for_claude(words)

    assert isinstance(output, str), "_format_transcript_for_claude must return a str"
    assert "probability" not in output, (
        "Output must not contain 'probability' — token-efficient format strips it"
    )
    # Should contain timestamp markers
    assert "0:00" in output or "00:00" in output or "[" in output, (
        "Output must contain timestamp markers like [M:SS]"
    )


# ---------------------------------------------------------------------------
# Test 8: _time_to_seconds parses MM:SS and HH:MM:SS
# ---------------------------------------------------------------------------

def test_time_to_seconds_formats():
    """_time_to_seconds must handle both MM:SS and HH:MM:SS formats."""
    from backend.services.clip_detector import _time_to_seconds

    assert _time_to_seconds("1:30") == 90.0, (
        f"'1:30' should parse to 90.0, got {_time_to_seconds('1:30')}"
    )
    assert _time_to_seconds("1:01:30") == 3690.0, (
        f"'1:01:30' should parse to 3690.0, got {_time_to_seconds('1:01:30')}"
    )
