"""
Tests for Task 2: Redis + RQ job processor scaffold with stale job recovery.
TDD RED phase — defines expected behavior before implementation.
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, Job


def _make_in_memory_db():
    """Helper: create an in-memory SQLite session with schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _create_job(db, status: str, minutes_old: int = 0) -> Job:
    """Helper: create a job with a specific status and age."""
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        youtube_url="https://youtube.com/watch?v=test123",
        status=status,
    )
    db.add(job)
    db.commit()

    if minutes_old > 0:
        # Backdate updated_at to simulate a stale job
        old_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_old)
        db.query(Job).filter(Job.id == job_id).update(
            {"updated_at": old_time},
            synchronize_session="fetch",
        )
        db.commit()

    db.refresh(job)
    return job


class TestRecoverStaleJobs:
    """Tests for the stale job recovery scan."""

    def test_stale_downloading_job_marked_failed(self):
        """A job in 'downloading' state older than 30 min must be marked failed."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "downloading", minutes_old=31)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "failed"
        assert "stale_recovery" in (job.error_message or "")

    def test_stale_transcribing_job_marked_failed(self):
        """A job in 'transcribing' state older than 30 min must be marked failed."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "transcribing", minutes_old=35)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "failed"
        assert "stale_recovery" in (job.error_message or "")

    def test_stale_analyzing_job_marked_failed(self):
        """A job in 'analyzing' state older than 30 min must be marked failed."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "analyzing", minutes_old=40)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "failed"
        assert "stale_recovery" in (job.error_message or "")

    def test_stale_generating_thumbnails_job_marked_failed(self):
        """A job in 'generating_thumbnails' older than 30 min must be marked failed."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "generating_thumbnails", minutes_old=31)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "failed"
        assert "stale_recovery" in (job.error_message or "")

    def test_fresh_job_not_affected(self):
        """A job in 'downloading' state that is only 5 min old must NOT be touched."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "downloading", minutes_old=5)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "downloading"  # unchanged

    def test_completed_job_not_affected(self):
        """A job in 'complete' status must never be touched by recovery scan."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "complete", minutes_old=60)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "complete"  # unchanged

    def test_pending_job_not_affected(self):
        """A job in 'pending' status that is old must NOT be touched (not transient)."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "pending", minutes_old=60)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "pending"  # unchanged

    def test_error_message_format(self):
        """Stale jobs must have error_message = 'stale_recovery: job abandoned'."""
        from backend.workers.job_processor import recover_stale_jobs

        db = _make_in_memory_db()
        job = _create_job(db, "transcribing", minutes_old=31)

        recover_stale_jobs(db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.error_message == "stale_recovery: job abandoned"


class TestProcessJob:
    """Tests for the process_job function structure."""

    def test_process_job_is_not_async(self):
        """process_job must be a regular (sync) function — RQ workers are not async."""
        import inspect
        from backend.workers.job_processor import process_job

        assert not inspect.iscoroutinefunction(process_job), \
            "process_job must be a sync function (not async) for RQ compatibility"

    def test_process_job_callable_with_rq_signature(self):
        """process_job must accept (job_id, youtube_url, db_path) positional args."""
        import inspect
        from backend.workers.job_processor import process_job

        sig = inspect.signature(process_job)
        params = list(sig.parameters.keys())

        assert "job_id" in params
        assert "youtube_url" in params
        assert "db_path" in params

    def test_process_job_sets_failed_on_not_implemented(self):
        """process_job must set status='failed' when a placeholder service raises NotImplementedError."""
        from backend.workers.job_processor import process_job

        db = _make_in_memory_db()
        job = _create_job(db, "pending")

        # process_job opens its own DB session from db_path, but we use override for test
        # It will fail at the first placeholder service (download_video NotImplementedError)
        process_job(job_id=job.id, youtube_url=job.youtube_url, db_path=":memory:", _db_override=db)

        db.refresh(job)
        assert job.status == "failed"
        assert job.error_message is not None
