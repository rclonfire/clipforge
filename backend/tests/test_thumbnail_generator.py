"""
Tests for Plan 01-04 Task 1: Claude frame selector + Pillow compositor.
TDD RED phase — defines expected behavior before implementation.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_frame(width: int = 512, height: int = 288) -> str:
    """Create a temporary JPEG frame file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(tmp.name, "JPEG")
    tmp.close()
    return tmp.name


def _make_frame_dict(frame_path: str, frame_index: int = 0) -> dict:
    """Build a frame dict as returned by extract_candidate_frames."""
    return {
        "frame_index": frame_index,
        "file_path": frame_path,
        "timestamp": frame_index * 5.0,
        "face_score": 0.8,
        "quality_score": 0.9,
        "combined_score": 0.85,
    }


def _make_concept(frame_index: int = 0) -> dict:
    """Build a minimal concept dict as Claude would return."""
    return {
        "frame_index": frame_index,
        "text_overlay": "TEST THUMBNAIL",
        "text_position": "bottom-left",
        "text_color": "#FFFFFF",
        "text_stroke_color": "#000000",
        "background_treatment": "none",
        "text_hierarchy": "large",
        "font_style": "condensed",
        "highlight_word": None,
        "highlight_color": None,
        "arrow_overlay": None,
        "emoji_overlays": None,
        "quote_style": False,
        "style_notes": "authentic",
        "reasoning": "test concept",
        "estimated_ctr_tier": "high",
    }


# ---------------------------------------------------------------------------
# Test: Claude model ID is not deprecated
# ---------------------------------------------------------------------------

def test_claude_model_is_not_deprecated():
    """analyze_frames_with_claude must use claude-haiku-4-5-20251001, NOT claude-3-haiku-20240307."""
    import inspect
    from backend.services import thumbnail_generator
    source = inspect.getsource(thumbnail_generator)
    assert "claude-3-haiku-20240307" not in source, (
        "Deprecated model claude-3-haiku-20240307 found in thumbnail_generator.py. "
        "Update to claude-haiku-4-5-20251001."
    )
    # Model used in analyze_frames_with_claude should be haiku, not sonnet
    assert "claude-haiku-4-5-20251001" in source, (
        "claude-haiku-4-5-20251001 must be used in analyze_frames_with_claude()"
    )


# ---------------------------------------------------------------------------
# Test: THUMBNAIL_SYSTEM_PROMPT uses selection, not generation
# ---------------------------------------------------------------------------

def test_thumbnail_prompt_instructs_selection_not_generation():
    """THUMBNAIL_SYSTEM_PROMPT must tell Claude to select frames, not generate images."""
    from backend.prompts.thumbnail_system import THUMBNAIL_SYSTEM_PROMPT
    prompt_lower = THUMBNAIL_SYSTEM_PROMPT.lower()
    # Must contain selection language
    assert "select" in prompt_lower, "Prompt must instruct Claude to SELECT frames"
    # Must NOT claim Claude generates new images
    assert "generate" not in prompt_lower or "you are not generating" in prompt_lower or \
           "selecting" in prompt_lower, (
        "Prompt must not describe Claude as generating images. "
        "Claude selects from existing frames."
    )


# ---------------------------------------------------------------------------
# Test: compose_thumbnail exists with correct signature
# ---------------------------------------------------------------------------

def test_compose_thumbnail_exists_and_returns_path():
    """compose_thumbnail(frame_path, concept, output_path) must exist and return a string path."""
    from backend.services.thumbnail_generator import compose_thumbnail

    frame_path = _make_test_frame()
    concept = _make_concept(frame_index=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_thumb.jpg")
        result = compose_thumbnail(frame_path, concept, output_path)

        assert isinstance(result, str), "compose_thumbnail must return a str path"
        assert Path(result).exists(), f"Output file must exist at {result}"

    os.unlink(frame_path)


def test_compose_thumbnail_produces_correct_dimensions():
    """compose_thumbnail must produce a 1280x720 JPEG."""
    from backend.services.thumbnail_generator import compose_thumbnail

    frame_path = _make_test_frame()
    concept = _make_concept(frame_index=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_thumb.jpg")
        result_path = compose_thumbnail(frame_path, concept, output_path)

        img = Image.open(result_path)
        assert img.size == (1280, 720), f"Expected 1280x720, got {img.size}"
        assert img.format == "JPEG", f"Expected JPEG format, got {img.format}"

    os.unlink(frame_path)


def test_compose_thumbnail_reads_from_existing_frame():
    """test_no_ai_generation_in_default_mode: compose_thumbnail must read an EXISTING frame file, never call any image generation API."""
    from backend.services.thumbnail_generator import compose_thumbnail

    frame_path = _make_test_frame()
    concept = _make_concept(frame_index=0)

    # If compose_thumbnail calls any AI generation API, it should raise since
    # we haven't mocked the API. With no mocks, it should succeed by using PIL only.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_thumb.jpg")
        # This must NOT import or call gemini_thumbnail or any generation model
        # We verify by checking: no RuntimeError about GEMINI_API_KEY is raised
        result_path = compose_thumbnail(frame_path, concept, output_path)
        assert Path(result_path).exists()

    os.unlink(frame_path)


# ---------------------------------------------------------------------------
# Test: 160x90 preview generated alongside thumbnail
# ---------------------------------------------------------------------------

def test_compose_thumbnail_generates_preview():
    """compose_thumbnail must save a 160x90 _preview.jpg alongside the main thumbnail."""
    from backend.services.thumbnail_generator import compose_thumbnail

    frame_path = _make_test_frame()
    concept = _make_concept(frame_index=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_thumb.jpg")
        result_path = compose_thumbnail(frame_path, concept, output_path)

        # Preview should be at same path but with _preview suffix
        preview_path = os.path.join(tmpdir, "test_thumb_preview.jpg")
        assert Path(preview_path).exists(), (
            f"160x90 preview file must exist at {preview_path}"
        )
        preview_img = Image.open(preview_path)
        assert preview_img.size == (160, 90), (
            f"Preview must be 160x90, got {preview_img.size}"
        )

    os.unlink(frame_path)


# ---------------------------------------------------------------------------
# Test: generate_thumbnails orchestrator
# ---------------------------------------------------------------------------

def test_generate_thumbnails_callable():
    """generate_thumbnails must be importable and callable."""
    from backend.services.thumbnail_generator import generate_thumbnails
    assert callable(generate_thumbnails)


def test_generate_thumbnails_with_mocked_claude():
    """generate_thumbnails(frames, transcript, job_id) -> list of dicts with file_path."""
    import backend.services.thumbnail_generator as tg_module
    from backend.services.thumbnail_generator import generate_thumbnails

    frame_path = _make_test_frame()
    frames = [_make_frame_dict(frame_path, i) for i in range(5)]
    transcript = {"text": "This is a test transcript about natural hair care routines."}

    mock_concept = _make_concept(frame_index=0)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps({"concepts": [mock_concept]})

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch THUMBNAILS_DIR on the module object directly so the function sees it
        original_thumbnails_dir = tg_module.THUMBNAILS_DIR
        tg_module.THUMBNAILS_DIR = Path(tmpdir)
        try:
            with patch("backend.services.thumbnail_generator.anthropic.Anthropic", return_value=mock_client):
                job_id = str(uuid.uuid4())
                results = generate_thumbnails(frames, transcript, job_id)
        finally:
            tg_module.THUMBNAILS_DIR = original_thumbnails_dir

        assert isinstance(results, list), "generate_thumbnails must return a list"
        assert len(results) >= 1, "Must produce at least 1 thumbnail"

        for item in results:
            assert "file_path" in item, "Each result must have file_path"
            assert Path(item["file_path"]).exists(), f"Thumbnail file must exist: {item['file_path']}"

    os.unlink(frame_path)


def test_generate_thumbnails_enhance_defaults_false():
    """generate_thumbnails must accept an enhance parameter defaulting to False."""
    import inspect
    from backend.services.thumbnail_generator import generate_thumbnails
    sig = inspect.signature(generate_thumbnails)
    params = sig.parameters
    assert "enhance" in params, "generate_thumbnails must have an 'enhance' parameter"
    assert params["enhance"].default == False, "enhance must default to False"
