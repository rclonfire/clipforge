"""
Tests for Plan 01-04 Task 2: Gemini enhancement as optional post-processing toggle.
TDD RED phase — defines expected behavior before implementation.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_thumbnail(width: int = 1280, height: int = 720) -> str:
    """Create a temporary 1280x720 JPEG thumbnail and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.new("RGB", (width, height), color=(80, 120, 160))
    img.save(tmp.name, "JPEG")
    tmp.close()
    return tmp.name


def _make_concept() -> dict:
    """Return a minimal concept dict."""
    return {
        "frame_index": 0,
        "text_overlay": "TEST THUMB",
        "text_position": "bottom-left",
        "text_color": "#FFFFFF",
        "text_stroke_color": "#000000",
        "style_notes": "authentic",
        "reasoning": "test",
        "estimated_ctr_tier": "high",
    }


# ---------------------------------------------------------------------------
# Test: enhance_thumbnail_with_gemini exists with correct signature
# ---------------------------------------------------------------------------

def test_enhance_thumbnail_with_gemini_is_importable():
    """enhance_thumbnail_with_gemini must be importable from gemini_thumbnail."""
    from backend.services.gemini_thumbnail import enhance_thumbnail_with_gemini
    assert callable(enhance_thumbnail_with_gemini)


# ---------------------------------------------------------------------------
# Test: USE_GEMINI_ENHANCEMENT defaults to False
# ---------------------------------------------------------------------------

def test_use_gemini_enhancement_defaults_false():
    """USE_GEMINI_ENHANCEMENT must default to False in code (a populated .env may enable it)."""
    from backend.config import settings, Settings
    assert hasattr(settings, "use_gemini_enhancement"), (
        "settings must have use_gemini_enhancement attribute"
    )
    # Assert the code default, independent of any local .env override.
    assert Settings.model_fields["use_gemini_enhancement"].default is False, (
        "USE_GEMINI_ENHANCEMENT must default to False"
    )


# ---------------------------------------------------------------------------
# Test: test_gemini_enhancement_is_optional
# When USE_GEMINI_ENHANCEMENT=False, generate_thumbnails does NOT call any gemini module
# ---------------------------------------------------------------------------

def test_gemini_enhancement_is_optional():
    """When enhance=False, generate_thumbnails must NOT import or call any gemini module."""
    import backend.services.thumbnail_generator as tg_module
    from backend.services.thumbnail_generator import generate_thumbnails
    import json

    # Create a real frame file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.new("RGB", (512, 288), color=(100, 150, 200))
    img.save(tmp.name, "JPEG")
    tmp.close()

    frames = [{"frame_index": 0, "file_path": tmp.name, "timestamp": 0.0,
               "face_score": 0.8, "quality_score": 0.9, "combined_score": 0.85}]
    concept = _make_concept()

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({"concepts": [concept]})
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    gemini_call_tracker = []

    def mock_enhance(thumbnail_path, concept, video_title):
        gemini_call_tracker.append(thumbnail_path)
        return thumbnail_path

    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = tg_module.THUMBNAILS_DIR
        tg_module.THUMBNAILS_DIR = Path(tmpdir)
        try:
            with patch("backend.services.thumbnail_generator.anthropic.Anthropic", return_value=mock_client):
                with patch("backend.services.gemini_thumbnail.enhance_thumbnail_with_gemini", side_effect=mock_enhance):
                    import uuid
                    results = generate_thumbnails(frames, {"text": "test"}, str(uuid.uuid4()), enhance=False)
        finally:
            tg_module.THUMBNAILS_DIR = original_dir

    assert len(gemini_call_tracker) == 0, (
        "enhance_thumbnail_with_gemini must NOT be called when enhance=False. "
        f"It was called {len(gemini_call_tracker)} times."
    )

    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test: test_gemini_receives_pillow_output
# When enhance=True, enhance_thumbnail_with_gemini receives the PILLOW output path
# ---------------------------------------------------------------------------

def test_gemini_receives_pillow_output():
    """When enhance=True, enhance_thumbnail_with_gemini must receive the Pillow JPG path."""
    import backend.services.thumbnail_generator as tg_module
    from backend.services.thumbnail_generator import generate_thumbnails
    import json

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.new("RGB", (512, 288), color=(100, 150, 200))
    img.save(tmp.name, "JPEG")
    tmp.close()

    frames = [{"frame_index": 0, "file_path": tmp.name, "timestamp": 0.0,
               "face_score": 0.8, "quality_score": 0.9, "combined_score": 0.85}]
    concept = _make_concept()

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({"concepts": [concept]})
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    gemini_received_paths = []

    def mock_enhance(thumbnail_path, concept, video_title):
        gemini_received_paths.append(thumbnail_path)
        return thumbnail_path  # return same path (simulated success)

    import backend.config as config_module

    tmpdir_obj = tempfile.mkdtemp()
    original_dir = tg_module.THUMBNAILS_DIR
    tg_module.THUMBNAILS_DIR = Path(tmpdir_obj)
    original_key = config_module.settings.gemini_api_key
    # Temporarily give settings a fake key so enhance branch is entered
    object.__setattr__(config_module.settings, "gemini_api_key", "fake-key")
    try:
        with patch("backend.services.thumbnail_generator.anthropic.Anthropic", return_value=mock_client):
            with patch("backend.services.gemini_thumbnail.enhance_thumbnail_with_gemini", side_effect=mock_enhance):
                import uuid
                results = generate_thumbnails(frames, {"text": "test"}, str(uuid.uuid4()), enhance=True)

        assert len(gemini_received_paths) >= 1, (
            "enhance_thumbnail_with_gemini must be called when enhance=True"
        )
        for path in gemini_received_paths:
            assert path.endswith(".jpg"), f"Gemini must receive a .jpg path, got: {path}"
            # The path must NOT be the original frame (512x288), it's the Pillow output
            img_check = Image.open(path)
            assert img_check.size == (1280, 720), (
                f"Gemini must receive a 1280x720 Pillow output, got {img_check.size}"
            )
    finally:
        tg_module.THUMBNAILS_DIR = original_dir
        object.__setattr__(config_module.settings, "gemini_api_key", original_key)

    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test: test_gemini_enhancement_fallback
# If Gemini raises RuntimeError, generate_thumbnails uses Pillow output unchanged
# ---------------------------------------------------------------------------

def test_gemini_enhancement_fallback():
    """If Gemini raises any exception, generate_thumbnails returns Pillow-only output gracefully."""
    import backend.services.thumbnail_generator as tg_module
    from backend.services.thumbnail_generator import generate_thumbnails
    import json

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.new("RGB", (512, 288), color=(100, 150, 200))
    img.save(tmp.name, "JPEG")
    tmp.close()

    frames = [{"frame_index": 0, "file_path": tmp.name, "timestamp": 0.0,
               "face_score": 0.8, "quality_score": 0.9, "combined_score": 0.85}]
    concept = _make_concept()

    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({"concepts": [concept]})
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    def mock_enhance_raises(thumbnail_path, concept, video_title):
        raise RuntimeError("GEMINI_API_KEY is not set")

    import backend.config as config_module

    tmpdir_obj = tempfile.mkdtemp()
    original_dir = tg_module.THUMBNAILS_DIR
    tg_module.THUMBNAILS_DIR = Path(tmpdir_obj)
    original_key = config_module.settings.gemini_api_key
    object.__setattr__(config_module.settings, "gemini_api_key", "fake-key")
    try:
        with patch("backend.services.thumbnail_generator.anthropic.Anthropic", return_value=mock_client):
            with patch("backend.services.gemini_thumbnail.enhance_thumbnail_with_gemini", side_effect=mock_enhance_raises):
                import uuid
                # Must NOT raise — should fall back gracefully
                results = generate_thumbnails(frames, {"text": "test"}, str(uuid.uuid4()), enhance=True)

        assert isinstance(results, list), "generate_thumbnails must return a list even when Gemini fails"
        assert len(results) >= 1, "Must still produce Pillow output when Gemini fails"
        for item in results:
            assert Path(item["file_path"]).exists(), (
                f"Pillow fallback thumbnail must still exist: {item['file_path']}"
            )
            assert item["generation_type"] == "pillow", (
                "generation_type must be 'pillow' when Gemini fallback occurs"
            )
    finally:
        tg_module.THUMBNAILS_DIR = original_dir
        object.__setattr__(config_module.settings, "gemini_api_key", original_key)

    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test: enhance_thumbnail_with_gemini accepts pillow thumbnail as input
# ---------------------------------------------------------------------------

def test_enhance_thumbnail_with_gemini_signature():
    """enhance_thumbnail_with_gemini must accept (thumbnail_path, concept, video_title)."""
    import inspect
    from backend.services.gemini_thumbnail import enhance_thumbnail_with_gemini
    sig = inspect.signature(enhance_thumbnail_with_gemini)
    params = list(sig.parameters.keys())
    assert "thumbnail_path" in params, "Must have thumbnail_path parameter"
    assert "concept" in params, "Must have concept parameter"
    assert "video_title" in params, "Must have video_title parameter"


def test_enhance_thumbnail_with_gemini_fallback_on_no_api_key():
    """enhance_thumbnail_with_gemini must return thumbnail_path unchanged when GEMINI_API_KEY is missing."""
    from backend.services.gemini_thumbnail import enhance_thumbnail_with_gemini

    thumb_path = _make_test_thumbnail()
    concept = _make_concept()

    # With no actual API key, it should fall back gracefully (not raise)
    with patch("backend.services.gemini_thumbnail.GEMINI_API_KEY", ""):
        result = enhance_thumbnail_with_gemini(thumb_path, concept, "Test Video")

    # Should return the original path unchanged (not crash)
    assert result == thumb_path or Path(result).exists(), (
        "Must return a valid path even when GEMINI_API_KEY is empty"
    )

    os.unlink(thumb_path)


# ---------------------------------------------------------------------------
# Test: original generate_thumbnails from gemini_thumbnail preserved as standalone
# ---------------------------------------------------------------------------

def test_generate_gemini_thumbnails_standalone_preserved():
    """generate_gemini_thumbnails_standalone must still exist in gemini_thumbnail.py."""
    from backend.services import gemini_thumbnail
    assert hasattr(gemini_thumbnail, "generate_gemini_thumbnails_standalone"), (
        "generate_gemini_thumbnails_standalone must be preserved in gemini_thumbnail.py "
        "(renamed from original generate_thumbnails)"
    )
    assert callable(gemini_thumbnail.generate_gemini_thumbnails_standalone)
