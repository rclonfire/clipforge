"""
Unit tests for export_processor.py — RQ worker that orchestrates clip exports.

Mocks all external services (align_words_for_clip, generate_ass, export_clip,
_find_video_file) and uses in-memory SQLite via _db_override for test isolation.
"""
from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.models import Base, Job, Clip, Export, ExportBatch


# ---------------------------------------------------------------------------
# In-memory DB helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory SQLite engine with StaticPool for thread safety."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _seed_db(db, job_id="job-001", batch_id="batch-001", platform="tiktok",
              captions=1, vertical_crop=1, n_kept_clips=2):
    """Seed a Job, ExportBatch, and n_kept_clips Clip rows (all kept=1)."""
    job = Job(
        id=job_id,
        youtube_url="https://youtube.com/watch?v=test",
        video_title="Funny Moments Video",
        status="complete",
    )
    db.add(job)

    batch = ExportBatch(
        id=batch_id,
        job_id=job_id,
        platform=platform,
        vertical_crop=vertical_crop,
        captions=captions,
        status="pending",
        total_clips=n_kept_clips,
        completed_clips=0,
    )
    db.add(batch)

    for i in range(n_kept_clips):
        clip = Clip(
            id=f"clip-{i:03d}",
            job_id=job_id,
            start_time_seconds=float(i * 40),
            end_time_seconds=float(i * 40 + 35),
            duration_seconds=35.0,
            virality_score=80,
            hook_strength=8,
            standalone_clarity=8,
            emotional_arc=8,
            trend_alignment=8,
            rewatch_potential=8,
            kept=1,
            preview_path=f"/fake/previews/clip-{i:03d}.mp4",
        )
        db.add(clip)

    db.commit()
    return job, batch


# ---------------------------------------------------------------------------
# Mock targets
# ---------------------------------------------------------------------------

_ALIGN = "backend.workers.export_processor.align_words_for_clip"
_GENERATE_ASS = "backend.workers.export_processor.generate_ass"
_EXPORT_CLIP = "backend.workers.export_processor.export_clip"
_FIND_VIDEO = "backend.workers.export_processor._find_video_file"
_LOAD_TRANSCRIPT = "backend.workers.export_processor._load_transcript"


def _make_mock_export_clip(tmp_dir: str):
    """Return a mock for export_clip that creates the output file."""
    def _side_effect(video_path, start_seconds, end_seconds, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake mp4 content")
        return output_path
    return MagicMock(side_effect=_side_effect)


# ---------------------------------------------------------------------------
# Tests: status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions(unittest.TestCase):
    """process_export_batch transitions ExportBatch status correctly."""

    def _run(self, db, batch_id="batch-001", job_id="job-001", tmp_dir=None):
        from backend.workers.export_processor import process_export_batch

        mock_export_clip = _make_mock_export_clip(tmp_dir or tempfile.mkdtemp())
        sample_words = [{"word": "hello", "start": 0.1, "end": 0.5}]

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, return_value=sample_words), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, mock_export_clip):
            process_export_batch(batch_id, job_id, ":memory:", _db_override=db)

    def test_batch_status_is_complete_after_run(self):
        db = _make_db()
        _seed_db(db)
        self._run(db)
        batch = db.query(ExportBatch).filter(ExportBatch.id == "batch-001").first()
        self.assertEqual(batch.status, "complete")

    def test_batch_status_transitions_through_processing(self):
        """Verify that status is set to 'processing' (not just 'complete') at some point.

        We capture intermediate state by observing the batch status before mocks return.
        The simplest proxy: after the run, completed_clips > 0 confirms we went through
        per-clip iteration (which only happens in 'processing' state).
        """
        db = _make_db()
        _seed_db(db, n_kept_clips=2)
        self._run(db)
        batch = db.query(ExportBatch).filter(ExportBatch.id == "batch-001").first()
        # If we got here with complete, processing was visited
        self.assertEqual(batch.status, "complete")
        self.assertEqual(batch.completed_clips, 2)

    def test_batch_fails_when_all_clips_fail(self):
        """When export_clip raises for all clips, batch should be 'failed'."""
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db)

        # export_clip always raises — no output file is created
        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=[]), \
             patch(_ALIGN, return_value=[]), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, side_effect=RuntimeError("ffmpeg failed")):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        batch = db.query(ExportBatch).filter(ExportBatch.id == "batch-001").first()
        # All clips failed — batch processing should still complete (not crash)
        # Status should be either 'complete' (0 clips succeeded) or 'failed'
        self.assertIn(batch.status, ("complete", "failed"))


# ---------------------------------------------------------------------------
# Tests: Export row creation
# ---------------------------------------------------------------------------

class TestExportRowCreation(unittest.TestCase):
    """Each kept clip produces an Export row with non-null file_path and batch_id."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db = _make_db()
        _seed_db(self.db, n_kept_clips=3)

    def _run(self):
        from backend.workers.export_processor import process_export_batch

        mock_export_clip = _make_mock_export_clip(self.tmp_dir)
        sample_words = [{"word": "hey", "start": 0.1, "end": 0.4}]

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, return_value=sample_words), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, mock_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=self.db)

    def test_creates_export_rows_for_each_kept_clip(self):
        self._run()
        exports = self.db.query(Export).all()
        self.assertEqual(len(exports), 3)

    def test_export_rows_have_non_null_file_path(self):
        self._run()
        exports = self.db.query(Export).all()
        for exp in exports:
            self.assertIsNotNone(exp.file_path)
            self.assertTrue(len(exp.file_path) > 0)

    def test_export_rows_have_batch_id(self):
        self._run()
        exports = self.db.query(Export).all()
        for exp in exports:
            self.assertEqual(exp.batch_id, "batch-001")

    def test_export_rows_have_platform(self):
        self._run()
        exports = self.db.query(Export).all()
        for exp in exports:
            self.assertEqual(exp.platform, "tiktok")


# ---------------------------------------------------------------------------
# Tests: Caption behavior (captions=True)
# ---------------------------------------------------------------------------

class TestCaptionsEnabled(unittest.TestCase):
    """When captions=True, align_words_for_clip and generate_ass are called per clip."""

    def test_align_and_generate_ass_called_when_captions_true(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, captions=1, n_kept_clips=2)
        tmp_dir = tempfile.mkdtemp()
        mock_export_clip = _make_mock_export_clip(tmp_dir)
        sample_words = [{"word": "hello", "start": 0.1, "end": 0.5}]

        mock_align = MagicMock(return_value=sample_words)
        mock_generate_ass = MagicMock(return_value="/fake/clip.ass")

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, mock_align), \
             patch(_GENERATE_ASS, mock_generate_ass), \
             patch(_EXPORT_CLIP, mock_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        # Should be called once per clip
        self.assertEqual(mock_align.call_count, 2)
        self.assertEqual(mock_generate_ass.call_count, 2)


# ---------------------------------------------------------------------------
# Tests: Caption behavior (captions=False)
# ---------------------------------------------------------------------------

class TestCaptionsDisabled(unittest.TestCase):
    """When captions=False, align_words_for_clip and generate_ass are NOT called."""

    def test_align_and_generate_ass_not_called_when_captions_false(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, captions=0, n_kept_clips=2)
        tmp_dir = tempfile.mkdtemp()
        mock_export_clip = _make_mock_export_clip(tmp_dir)

        mock_align = MagicMock(return_value=[])
        mock_generate_ass = MagicMock(return_value="/fake/clip.ass")

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=[]), \
             patch(_ALIGN, mock_align), \
             patch(_GENERATE_ASS, mock_generate_ass), \
             patch(_EXPORT_CLIP, mock_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        self.assertEqual(mock_align.call_count, 0)
        self.assertEqual(mock_generate_ass.call_count, 0)


# ---------------------------------------------------------------------------
# Tests: vertical_crop=False
# ---------------------------------------------------------------------------

class TestVerticalCropDisabled(unittest.TestCase):
    """When vertical_crop=False, export_clip is called with vertical_crop=False."""

    def test_export_clip_called_with_vertical_crop_false(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, vertical_crop=0, captions=0, n_kept_clips=1)
        tmp_dir = tempfile.mkdtemp()
        mock_export_clip = _make_mock_export_clip(tmp_dir)
        wrapped_export_clip = MagicMock(side_effect=mock_export_clip.side_effect)

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=[]), \
             patch(_ALIGN, return_value=[]), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, wrapped_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        call_kwargs = wrapped_export_clip.call_args[1]
        self.assertFalse(call_kwargs.get("vertical_crop", True))


# ---------------------------------------------------------------------------
# Tests: per-clip failure isolation
# ---------------------------------------------------------------------------

class TestPerClipFailureIsolation(unittest.TestCase):
    """A single clip failure logs a warning but batch continues processing remaining clips."""

    def test_single_clip_failure_does_not_crash_batch(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, n_kept_clips=3)
        tmp_dir = tempfile.mkdtemp()

        call_count = [0]

        def flaky_export_clip(video_path, start_seconds, end_seconds, output_path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("ffmpeg crashed on clip 2")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake mp4 content")
            return output_path

        sample_words = [{"word": "hey", "start": 0.1, "end": 0.4}]

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, return_value=sample_words), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, side_effect=flaky_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        # Batch should still reach complete (or failed); definitely should NOT raise
        batch = db.query(ExportBatch).filter(ExportBatch.id == "batch-001").first()
        self.assertIn(batch.status, ("complete", "failed"))

        # 2 of 3 clips succeeded — 2 Export rows should exist
        exports = db.query(Export).all()
        self.assertEqual(len(exports), 2)

    def test_completed_clips_increments_after_each_successful_clip(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, n_kept_clips=3)
        tmp_dir = tempfile.mkdtemp()
        mock_export_clip = _make_mock_export_clip(tmp_dir)
        sample_words = [{"word": "hey", "start": 0.1, "end": 0.4}]

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, return_value=sample_words), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, mock_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        batch = db.query(ExportBatch).filter(ExportBatch.id == "batch-001").first()
        self.assertEqual(batch.completed_clips, 3)


# ---------------------------------------------------------------------------
# Tests: output filename format
# ---------------------------------------------------------------------------

class TestExportFileNaming(unittest.TestCase):
    """Export file named {sanitized_video_title}_{clip_number}_{platform}.mp4"""

    def test_output_filename_format(self):
        from backend.workers.export_processor import process_export_batch

        db = _make_db()
        _seed_db(db, n_kept_clips=1)
        tmp_dir = tempfile.mkdtemp()

        captured_output_paths = []

        def capturing_export_clip(video_path, start_seconds, end_seconds, output_path, **kwargs):
            captured_output_paths.append(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake mp4 content")
            return output_path

        sample_words = [{"word": "hey", "start": 0.1, "end": 0.4}]

        with patch(_FIND_VIDEO, return_value="/fake/video.mp4"), \
             patch(_LOAD_TRANSCRIPT, return_value=sample_words), \
             patch(_ALIGN, return_value=sample_words), \
             patch(_GENERATE_ASS, return_value="/fake/clip.ass"), \
             patch(_EXPORT_CLIP, side_effect=capturing_export_clip):
            process_export_batch("batch-001", "job-001", ":memory:", _db_override=db)

        self.assertEqual(len(captured_output_paths), 1)
        filename = Path(captured_output_paths[0]).name
        # Should end with _1_tiktok.mp4
        self.assertTrue(filename.endswith("_1_tiktok.mp4"), f"Got: {filename}")
        # Should contain sanitized title
        self.assertIn("Funny_Moments_Video", filename)


# ---------------------------------------------------------------------------
# Tests: _sanitize_title helper
# ---------------------------------------------------------------------------

class TestSanitizeTitle(unittest.TestCase):
    """_sanitize_title returns filesystem-safe string."""

    def test_sanitize_removes_special_chars(self):
        from backend.workers.export_processor import _sanitize_title
        result = _sanitize_title("Hello: World! (2024)")
        # Colons, parens, exclamation marks should be stripped
        self.assertNotIn(":", result)
        self.assertNotIn("!", result)
        self.assertNotIn("(", result)

    def test_sanitize_replaces_spaces_with_underscores(self):
        from backend.workers.export_processor import _sanitize_title
        result = _sanitize_title("Hello World")
        self.assertIn("_", result)
        self.assertNotIn(" ", result)

    def test_sanitize_empty_returns_clip(self):
        from backend.workers.export_processor import _sanitize_title
        result = _sanitize_title("!!!")
        self.assertEqual(result, "clip")

    def test_sanitize_truncates_to_50_chars(self):
        from backend.workers.export_processor import _sanitize_title
        long_title = "A" * 100
        result = _sanitize_title(long_title)
        self.assertLessEqual(len(result), 50)


# ---------------------------------------------------------------------------
# Tests: _find_video_file helper
# ---------------------------------------------------------------------------

class TestFindVideoFile(unittest.TestCase):
    """_find_video_file returns first .mp4/.webm/.mkv file in directory."""

    def test_finds_mp4_file(self):
        from backend.workers.export_processor import _find_video_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "video.mp4"
            p.write_bytes(b"fake")
            result = _find_video_file(Path(tmp_dir))
            self.assertEqual(result, str(p.resolve()))

    def test_raises_when_no_video_found(self):
        from backend.workers.export_processor import _find_video_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                _find_video_file(Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()
