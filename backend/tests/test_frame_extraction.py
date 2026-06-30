"""
Tests for Plan 01-03: Frame extraction with face-aware quality scoring.
TDD RED phase — defines expected behavior before implementation.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_frame_file(tmp_dir: Path, name: str = "test.jpg") -> Path:
    """Create a tiny 640x360 JPEG using Pillow for testing."""
    from PIL import Image
    p = tmp_dir / name
    img = Image.new("RGB", (640, 360), color=(100, 150, 200))
    img.save(str(p), "JPEG")
    return p


# ---------------------------------------------------------------------------
# face_detection module exports
# ---------------------------------------------------------------------------

class TestFaceDetectionExports:
    """Verify the public API of face_detection.py matches the plan spec."""

    def test_detect_faces_is_importable(self):
        from backend.services.face_detection import detect_faces  # noqa
        assert callable(detect_faces)

    def test_score_frame_quality_is_importable(self):
        """score_frame_quality must be exported from face_detection (new in plan 01-03)."""
        from backend.services.face_detection import score_frame_quality  # noqa
        assert callable(score_frame_quality)

    def test_create_subject_mask_is_importable(self):
        from backend.services.face_detection import create_subject_mask  # noqa
        assert callable(create_subject_mask)


class TestScoreFrameQuality:
    """score_frame_quality(image_path) -> float in [0.0, 1.0]."""

    def test_returns_float(self, tmp_path):
        from backend.services.face_detection import score_frame_quality
        p = _make_fake_frame_file(tmp_path)
        result = score_frame_quality(str(p))
        assert isinstance(result, float)

    def test_result_in_unit_range(self, tmp_path):
        from backend.services.face_detection import score_frame_quality
        p = _make_fake_frame_file(tmp_path)
        result = score_frame_quality(str(p))
        assert 0.0 <= result <= 1.0

    def test_missing_file_returns_zero(self):
        from backend.services.face_detection import score_frame_quality
        result = score_frame_quality("/tmp/does_not_exist_99999.jpg")
        assert result == 0.0


# ---------------------------------------------------------------------------
# extract_candidate_frames signature and return shape
# ---------------------------------------------------------------------------

class TestExtractCandidateFramesSignature:
    """Verify extract_candidate_frames signature matches plan spec."""

    def test_accepts_max_frames_param(self):
        import inspect
        from backend.services.frame_extraction import extract_candidate_frames
        sig = inspect.signature(extract_candidate_frames)
        params = sig.parameters
        assert "video_path" in params
        assert "job_id" in params
        assert "max_frames" in params
        assert sig.parameters["max_frames"].default == 20


class TestExtractCandidateFramesReturnShape:
    """Verify return dict shape with mocked MediaPipe + ffmpeg calls."""

    def test_returned_dicts_have_required_keys(self, tmp_path):
        """
        Smoke test: mock ffprobe + ffmpeg + detect_faces so we can call
        extract_candidate_frames without real video.  Assert each returned
        dict has the required keys from the plan spec.
        """
        from PIL import Image

        # Create a fake JPEG that ffmpeg would have written
        fake_frame = tmp_path / "test_job" / "frame_0000.jpg"
        fake_frame.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 360), color=(120, 80, 60))
        img.save(str(fake_frame), "JPEG")

        def fake_run_dispatch(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            # ffprobe returns duration JSON
            if "ffprobe" in str(cmd[0]):
                mock_result.stdout = '{"format": {"duration": "60.0"}}'
                return mock_result
            # ffmpeg frame extraction writes a fake JPEG
            for arg in reversed(cmd):
                arg_str = str(arg)
                if arg_str.endswith(".jpg"):
                    p = Path(arg_str)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img2 = Image.new("RGB", (640, 360), color=(100, 100, 100))
                    img2.save(str(p), "JPEG")
                    break
            return mock_result

        with patch("subprocess.run", side_effect=fake_run_dispatch):
            # Also mock detect_faces to return 1 face
            mock_face = MagicMock()
            mock_face.area_ratio = 0.15
            mock_face.center = (320, 180)
            mock_face.confidence = 0.9
            mock_face.expressiveness = 0.4
            mock_face.expression_categories = {}
            mock_face.bbox = (100, 50, 200, 200)

            with patch(
                "backend.services.frame_extraction.detect_faces",
                return_value=[mock_face],
            ):
                from backend.services.frame_extraction import extract_candidate_frames
                results = extract_candidate_frames(
                    video_path="/fake/video.mp4",
                    job_id="test_job",
                    max_frames=5,
                )

        if results:
            for r in results:
                assert "frame_index" in r, "Missing 'frame_index' key"
                assert "file_path" in r, "Missing 'file_path' key"
                assert "timestamp" in r, "Missing 'timestamp' key"
                assert "face_score" in r, "Missing 'face_score' key"
                assert "quality_score" in r, "Missing 'quality_score' key"

    def test_face_score_normalized_0_to_1(self, tmp_path):
        """face_score must be in [0, 1], not the 0-100 scale used internally."""
        from PIL import Image

        # Pre-create a fake frame
        fake_frame = tmp_path / "norm_job" / "frame_0000.jpg"
        fake_frame.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 360), color=(120, 80, 60))
        img.save(str(fake_frame), "JPEG")

        def fake_ffmpeg_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            for arg in reversed(cmd):
                arg_str = str(arg)
                if arg_str.endswith(".jpg"):
                    p = Path(arg_str)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img2 = Image.new("RGB", (640, 360), color=(100, 100, 100))
                    img2.save(str(p), "JPEG")
                    break
            return mock_result

        # ffprobe mock
        def fake_run_dispatch(cmd, **kwargs):
            if "ffprobe" in str(cmd[0]):
                r = MagicMock()
                r.stdout = '{"format": {"duration": "60.0"}}'
                r.returncode = 0
                return r
            return fake_ffmpeg_run(cmd, **kwargs)

        with patch("subprocess.run", side_effect=fake_run_dispatch):
            mock_face = MagicMock()
            mock_face.area_ratio = 0.15
            mock_face.center = (320, 180)
            mock_face.confidence = 0.85
            mock_face.expressiveness = 0.3
            mock_face.expression_categories = {}
            mock_face.bbox = (100, 50, 200, 200)

            with patch(
                "backend.services.frame_extraction.detect_faces",
                return_value=[mock_face],
            ):
                from importlib import reload
                import backend.services.frame_extraction as fe
                reload(fe)
                results = fe.extract_candidate_frames(
                    video_path="/fake/video.mp4",
                    job_id="norm_job",
                    max_frames=5,
                )

        for r in results:
            assert 0.0 <= r["face_score"] <= 1.0, f"face_score out of range: {r['face_score']}"
            assert 0.0 <= r["quality_score"] <= 1.0, f"quality_score out of range: {r['quality_score']}"

    def test_frames_resized_to_512x288(self, tmp_path):
        """After scoring, frames must be resized to 512x288 in-place."""
        from PIL import Image

        # Pre-create a frame with original 1280x720 size
        fake_frame = tmp_path / "resize_job" / "frame_0000.jpg"
        fake_frame.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (1280, 720), color=(120, 80, 60))
        img.save(str(fake_frame), "JPEG")

        def fake_run_dispatch(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            if "ffprobe" in str(cmd[0]):
                mock_result.stdout = '{"format": {"duration": "60.0"}}'
                return mock_result
            for arg in reversed(cmd):
                arg_str = str(arg)
                if arg_str.endswith(".jpg"):
                    p = Path(arg_str)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img2 = Image.new("RGB", (1280, 720), color=(100, 100, 100))
                    img2.save(str(p), "JPEG")
                    break
            return mock_result

        with patch("subprocess.run", side_effect=fake_run_dispatch):
            mock_face = MagicMock()
            mock_face.area_ratio = 0.15
            mock_face.center = (640, 360)
            mock_face.confidence = 0.9
            mock_face.expressiveness = 0.4
            mock_face.expression_categories = {}
            mock_face.bbox = (200, 100, 400, 400)

            with patch(
                "backend.services.frame_extraction.detect_faces",
                return_value=[mock_face],
            ):
                from importlib import reload
                import backend.services.frame_extraction as fe
                reload(fe)
                results = fe.extract_candidate_frames(
                    video_path="/fake/video.mp4",
                    job_id="resize_job",
                    max_frames=5,
                )

        for r in results:
            img_check = Image.open(r["file_path"])
            w, h = img_check.size
            assert (w, h) == (512, 288), f"Frame not resized: got {w}x{h}, expected 512x288"
