"""Tests for local-file ingestion (ingest_local_file + process_local_job)."""
from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, Job


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _job(db, status="pending"):
    job = Job(id=str(uuid.uuid4())[:12], youtube_url="/local/path/video.mp4", status=status)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


# ---------------------------------------------------------------------------
# ingest_local_file
# ---------------------------------------------------------------------------

def test_ingest_missing_file_raises():
    from backend.services.video_ingestion import ingest_local_file, IngestError

    with pytest.raises(IngestError) as exc:
        ingest_local_file("/no/such/file.mp4", "job-1")
    assert exc.value.error_code == "file_not_found"


def test_ingest_returns_pipeline_shape(tmp_path, monkeypatch):
    from backend.services import video_ingestion

    src = tmp_path / "MyEdit.mov"
    src.write_bytes(b"fake source video")
    monkeypatch.setattr(video_ingestion, "DOWNLOADS_DIR", tmp_path / "downloads")

    ok = MagicMock(returncode=0, stderr="", stdout="")
    with patch("subprocess.run", return_value=ok), patch.object(
        video_ingestion, "get_video_metadata", return_value={"format": {"duration": "42.5"}}
    ):
        result = video_ingestion.ingest_local_file(str(src), "job-2")

    assert set(result) == {"video_path", "title", "duration_seconds", "thumbnail_url"}
    assert result["title"] == "MyEdit"          # derived from filename stem
    assert result["duration_seconds"] == 42      # int(float("42.5"))
    assert result["video_path"].endswith("video.mp4")


def test_ingest_reencodes_when_stream_copy_fails(tmp_path, monkeypatch):
    from backend.services import video_ingestion

    src = tmp_path / "weird.avi"
    src.write_bytes(b"fake")
    monkeypatch.setattr(video_ingestion, "DOWNLOADS_DIR", tmp_path / "downloads")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # First call (stream copy) fails, second (re-encode) succeeds
        return MagicMock(returncode=(1 if len(calls) == 1 else 0), stderr="codec err", stdout="")

    with patch("subprocess.run", side_effect=fake_run), patch.object(
        video_ingestion, "get_video_metadata", return_value={"format": {"duration": "10"}}
    ):
        result = video_ingestion.ingest_local_file(str(src), "job-3")

    assert len(calls) == 2  # fell back to re-encode
    assert any("libx264" in " ".join(str(a) for a in c) for c in calls)
    assert result["duration_seconds"] == 10


# ---------------------------------------------------------------------------
# process_local_job
# ---------------------------------------------------------------------------

def test_process_local_job_signature_and_sync():
    from backend.workers.job_processor import process_local_job

    assert not inspect.iscoroutinefunction(process_local_job)
    params = list(inspect.signature(process_local_job).parameters.keys())
    assert "job_id" in params and "source_path" in params and "db_path" in params


def test_process_local_job_marks_failed_on_missing_file():
    from backend.workers.job_processor import process_local_job

    db = _db()
    job = _job(db)

    process_local_job(job_id=job.id, source_path="/no/such/file.mp4", db_path=":memory:", _db_override=db)

    db.refresh(job)
    assert job.status == "failed"
    assert job.error_message is not None
