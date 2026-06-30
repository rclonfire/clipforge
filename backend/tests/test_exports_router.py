"""
Tests for the exports router, clip kept state, and thumbnail download endpoint.

Uses TestClient with an in-memory SQLite database injected via dependency override.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.models import Base, Job, Clip, Thumbnail, Export, ExportBatch, get_db

# ---------------------------------------------------------------------------
# In-memory DB fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture()
def db_session():
    # StaticPool ensures the same in-memory DB connection is reused across
    # threads — required because TestClient runs requests in a separate thread.
    engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(db_session, job_id: str = "job-001") -> Job:
    job = Job(id=job_id, youtube_url="https://youtube.com/watch?v=test", status="complete")
    db_session.add(job)
    db_session.commit()
    return job


def _make_clip(db_session, job_id: str = "job-001", clip_id: str = "clip-001", kept: int = 0, preview_path: str | None = None) -> Clip:
    clip = Clip(
        id=clip_id,
        job_id=job_id,
        start_time_seconds=0.0,
        end_time_seconds=30.0,
        duration_seconds=30.0,
        virality_score=80,
        hook_strength=8,
        standalone_clarity=8,
        emotional_arc=8,
        trend_alignment=8,
        rewatch_potential=8,
        kept=kept,
        preview_path=preview_path,
    )
    db_session.add(clip)
    db_session.commit()
    return clip


# ---------------------------------------------------------------------------
# Test: ClipResponse includes `kept` field
# ---------------------------------------------------------------------------

def test_list_clips_includes_kept_field(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=1)

    response = client.get("/api/jobs/job-001/clips")
    assert response.status_code == 200
    clips = response.json()
    assert len(clips) == 1
    assert "kept" in clips[0]
    assert clips[0]["kept"] is True


def test_list_clips_kept_defaults_false(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=0)

    response = client.get("/api/jobs/job-001/clips")
    assert response.status_code == 200
    clips = response.json()
    assert clips[0]["kept"] is False


# ---------------------------------------------------------------------------
# Test: PATCH /clips/{clip_id} updates kept state
# ---------------------------------------------------------------------------

def test_patch_clip_kept_true(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=0)

    response = client.patch("/api/jobs/job-001/clips/clip-001", json={"kept": True})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "clip-001"
    assert data["kept"] is True


def test_patch_clip_kept_false(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=1)

    response = client.patch("/api/jobs/job-001/clips/clip-001", json={"kept": False})
    assert response.status_code == 200
    assert response.json()["kept"] is False


def test_patch_clip_not_found(client, db_session):
    _make_job(db_session)
    response = client.patch("/api/jobs/job-001/clips/nonexistent", json={"kept": True})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Test: POST /exports with no kept clips returns 400
# ---------------------------------------------------------------------------

def test_create_export_no_kept_clips(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=0)  # Not kept

    response = client.post("/api/jobs/job-001/exports", json={"platform": "tiktok"})
    assert response.status_code == 400
    assert "No kept clips" in response.json()["detail"]


def test_create_export_job_not_found(client, db_session):
    response = client.post("/api/jobs/nonexistent/exports", json={"platform": "tiktok"})
    assert response.status_code == 404


def test_create_export_with_kept_clips(client, db_session):
    _make_job(db_session)
    _make_clip(db_session, kept=1)

    # Mock _enqueue_export to avoid actual worker/thread creation
    with patch("backend.routers.exports._enqueue_export"):
        response = client.post("/api/jobs/job-001/exports", json={"platform": "tiktok"})

    assert response.status_code == 200
    data = response.json()
    assert "batch_id" in data
    assert data["status"] == "pending"
    assert data["total_clips"] == 1


# ---------------------------------------------------------------------------
# Test: GET /exports/{batch_id}/status returns ExportStatusResponse shape
# ---------------------------------------------------------------------------

def test_export_status(client, db_session):
    _make_job(db_session)
    batch = ExportBatch(
        id="batch-001",
        job_id="job-001",
        platform="tiktok",
        status="processing",
        progress_message="Processing...",
        total_clips=3,
        completed_clips=1,
    )
    db_session.add(batch)
    db_session.commit()

    response = client.get("/api/jobs/job-001/exports/batch-001/status")
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "batch-001"
    assert data["status"] == "processing"
    assert data["total_clips"] == 3
    assert data["completed_clips"] == 1
    assert data["progress_message"] == "Processing..."


def test_export_status_not_found(client, db_session):
    _make_job(db_session)
    response = client.get("/api/jobs/job-001/exports/nonexistent/status")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Test: GET /exports/{batch_id}/download returns ZIP content-type when complete
# ---------------------------------------------------------------------------

def test_download_export_zip_not_complete(client, db_session):
    _make_job(db_session)
    batch = ExportBatch(
        id="batch-001",
        job_id="job-001",
        platform="tiktok",
        status="processing",
        total_clips=1,
        completed_clips=0,
    )
    db_session.add(batch)
    db_session.commit()

    response = client.get("/api/jobs/job-001/exports/batch-001/download")
    assert response.status_code == 400
    assert "not yet complete" in response.json()["detail"]


def test_download_export_zip_complete(client, db_session, tmp_path):
    _make_job(db_session)
    clip = _make_clip(db_session, kept=1)

    # Create a real mp4 file stub
    fake_video = tmp_path / "clip-001.mp4"
    fake_video.write_bytes(b"fake mp4 content")

    batch = ExportBatch(
        id="batch-001",
        job_id="job-001",
        platform="tiktok",
        status="complete",
        total_clips=1,
        completed_clips=1,
    )
    db_session.add(batch)

    export = Export(
        id="exp-001",
        clip_id="clip-001",
        batch_id="batch-001",
        platform="tiktok",
        file_path=str(fake_video),
    )
    db_session.add(export)
    db_session.commit()

    response = client.get("/api/jobs/job-001/exports/batch-001/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]

    # Verify it's a valid ZIP
    buf = io.BytesIO(response.content)
    with zipfile.ZipFile(buf, "r") as zf:
        assert "clip-001.mp4" in zf.namelist()


# ---------------------------------------------------------------------------
# Test: Thumbnail download endpoint returns FileResponse with Content-Disposition
# ---------------------------------------------------------------------------

def test_thumbnail_download_not_found_job(client, db_session):
    response = client.get("/api/jobs/nonexistent/clips/clip-001/thumbnail")
    assert response.status_code == 404


def test_thumbnail_download_no_thumbnails(client, db_session):
    _make_job(db_session)
    _make_clip(db_session)

    response = client.get("/api/jobs/job-001/clips/clip-001/thumbnail")
    assert response.status_code == 404
    assert "thumbnails" in response.json()["detail"].lower()


def test_thumbnail_download_file_not_found(client, db_session):
    _make_job(db_session)
    _make_clip(db_session)

    thumb = Thumbnail(
        id="thumb-001",
        job_id="job-001",
        file_path="/nonexistent/path/thumb.jpg",
    )
    db_session.add(thumb)
    db_session.commit()

    response = client.get("/api/jobs/job-001/clips/clip-001/thumbnail")
    assert response.status_code == 404
    assert "file not found" in response.json()["detail"].lower()


def test_thumbnail_download_success(client, db_session, tmp_path):
    _make_job(db_session)
    _make_clip(db_session)

    fake_jpg = tmp_path / "thumb.jpg"
    fake_jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # JPEG magic bytes

    thumb = Thumbnail(
        id="thumb-001",
        job_id="job-001",
        file_path=str(fake_jpg),
    )
    db_session.add(thumb)
    db_session.commit()

    response = client.get("/api/jobs/job-001/clips/clip-001/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "Content-Disposition" in response.headers or "content-disposition" in response.headers
    disposition = response.headers.get("content-disposition") or response.headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert "thumbnail_job-001.jpg" in disposition
