"""
Jobs router — submit and track video processing jobs.

Job submission uses RQ when Redis is available. Falls back to a daemon thread
in development mode when Redis is not running. Production must have Redis running.
"""
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Job, get_db, SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# YouTube URL pattern — matches youtube.com/watch?v=... and youtu.be/... variants
_YOUTUBE_URL_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+",
    re.IGNORECASE,
)


class JobCreate(BaseModel):
    youtube_url: str

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        if not _YOUTUBE_URL_RE.match(v.strip()):
            raise ValueError(
                "URL must be a valid YouTube URL (youtube.com or youtu.be)"
            )
        return v.strip()


class JobResponse(BaseModel):
    id: str
    youtube_url: str
    video_title: Optional[str] = None
    video_duration_seconds: Optional[int] = None
    status: str
    progress_message: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


# Video file extensions accepted for local-path ingestion
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def _enqueue(func, job_id: str, payload: str) -> None:
    """
    Enqueue a processing function (process_job / process_local_job) via RQ if
    Redis is available. Falls back to a daemon thread (dev mode only) otherwise.

    func is called as func(job_id, payload, db_path); the daemon path injects a
    thread-local DB session.
    """
    db_path = str(settings.db_path)

    try:
        import redis as redis_lib
        from rq import Queue

        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()

        q = Queue(connection=r)
        q.enqueue(
            func,
            job_id,
            payload,
            db_path,
            result_ttl=86400,    # Keep result 24h
            failure_ttl=604800,  # Keep failure 7 days
        )
        logger.info(f"[{job_id}] Enqueued {func.__name__} via RQ on {settings.redis_url}")

    except Exception as exc:
        logger.warning(
            f"[{job_id}] Redis unavailable ({exc}) — falling back to daemon thread (dev mode only). "
            "Start Redis and an RQ worker for production use."
        )

        def _run():
            thread_db = SessionLocal()
            try:
                func(job_id, payload, db_path, _db_override=thread_db)
            finally:
                thread_db.close()

        thread = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
        thread.start()


@router.post("", response_model=JobResponse)
def create_job(body: JobCreate, db: Session = Depends(get_db)):
    """Submit a YouTube URL for processing."""
    job_id = str(uuid.uuid4())[:12]

    job = Job(
        id=job_id,
        youtube_url=body.youtube_url,
        status="pending",
        progress_message="Job queued, starting soon...",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from backend.workers.job_processor import process_job
    _enqueue(process_job, job_id, body.youtube_url)

    return _job_to_response(job)


class LocalJobCreate(BaseModel):
    source_path: str


@router.post("/local", response_model=JobResponse)
def create_local_job(body: LocalJobCreate, db: Session = Depends(get_db)):
    """Submit a local video file (e.g. an editor export) for processing by path."""
    src = Path(body.source_path).expanduser()
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {body.source_path}")
    if src.suffix.lower() not in _VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{src.suffix}'. Accepted: {', '.join(sorted(_VIDEO_EXTS))}",
        )

    job_id = str(uuid.uuid4())[:12]
    resolved = str(src.resolve())

    # Job.youtube_url is NOT NULL; for local jobs it records the source path.
    job = Job(
        id=job_id,
        youtube_url=resolved,
        status="pending",
        progress_message="Local file queued, starting soon...",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from backend.workers.job_processor import process_local_job
    _enqueue(process_local_job, job_id, resolved)

    return _job_to_response(job)


@router.get("", response_model=List[JobResponse])
def list_jobs(db: Session = Depends(get_db)):
    """List all jobs, newest first."""
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get a specific job's status and details."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


class JobProgressResponse(BaseModel):
    stage: str
    message: str


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str, db: Session = Depends(get_db)):
    """Get the current processing stage and progress message for a job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobProgressResponse(
        stage=job.status or "pending",
        message=job.progress_message or "",
    )


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        youtube_url=job.youtube_url,
        video_title=job.video_title,
        video_duration_seconds=job.video_duration_seconds,
        status=job.status,
        progress_message=job.progress_message,
        error_message=job.error_message,
        created_at=str(job.created_at) if job.created_at else None,
        completed_at=str(job.completed_at) if job.completed_at else None,
    )
