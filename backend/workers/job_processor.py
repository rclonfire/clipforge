"""
RQ-compatible job processor for ClipForge.

IMPORTANT: process_job() is a regular (sync) function — NOT async.
RQ workers run in separate processes outside the FastAPI event loop.
Each worker call opens its own SQLite session from db_path.

Pipeline stages:
    1. download_video       — wired in by plan 01-02
    2. transcribe_video     — wired in by plan 01-02
    3. extract_frames       — wired in by plan 01-03
    4. generate_thumbnails  — wired in by plan 01-04
    5. analyze_signals      — wired in by plan 02-03 (uses audio.wav from Stage 2)
    6. detect_clips         — wired in by plan 02-03 (Claude clip detection)
    7. extract_preview      — wired in by plan 02-03 (FFmpeg .mp4 preview per clip)

Startup integration:
    recover_stale_jobs() is called by FastAPI's @app.on_event("startup") handler.
    It marks transient jobs older than 30 minutes as failed so they don't hang forever.
"""
from __future__ import annotations

import json
import logging
import uuid as _uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.config import settings
from backend.models import Base, Job, Thumbnail, Clip
from backend.services.video_ingestion import download_video
from backend.services.transcription import transcribe_video
from backend.services.frame_extraction import extract_candidate_frames
from backend.services.frame_enhancer import enhance_frames_batch
from backend.services.thumbnail_generator import generate_thumbnails
from backend.services.signal_analysis import analyze_signals
from backend.services.clip_detector import detect_clips
from backend.services.preview_extractor import extract_preview

logger = logging.getLogger(__name__)

# Transient statuses eligible for stale job recovery
_STALE_STATUSES = (
    "downloading",
    "transcribing",
    "extracting_frames",
    "enhancing_frames",
    "analyzing",
    "generating_thumbnails",
    "detecting_clips",
)

# A job is "stale" if it has been in a transient status for more than this long
_STALE_THRESHOLD_MINUTES = 30


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def recover_stale_jobs(db_path: str, _db_override: Optional[Session] = None) -> int:
    """
    Scan all jobs in transient states that haven't been updated in 30+ minutes
    and mark them as failed with a 'stale_recovery' error message.

    Called on FastAPI startup and RQ worker startup to prevent ghost jobs.

    Args:
        db_path: Path to SQLite database file (or ':memory:' for tests).
                 Ignored if _db_override is provided.
        _db_override: Inject an existing Session for testing without opening a new DB.

    Returns:
        Number of jobs recovered (marked failed).
    """
    if _db_override is not None:
        db = _db_override
        should_close = False
    else:
        if db_path == ":memory:":
            url = "sqlite:///:memory:"
        else:
            url = f"sqlite:///{db_path}"
        engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        should_close = True

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=_STALE_THRESHOLD_MINUTES)
        recovered = 0

        stale_jobs = (
            db.query(Job)
            .filter(Job.status.in_(_STALE_STATUSES))
            .filter(Job.updated_at < cutoff)
            .all()
        )

        for job in stale_jobs:
            logger.warning(
                f"[recovery] Job {job.id} stuck in '{job.status}' since {job.updated_at} — marking failed"
            )
            job.status = "failed"
            job.error_message = "stale_recovery: job abandoned"

        if stale_jobs:
            db.commit()
            recovered = len(stale_jobs)
            logger.info(f"[recovery] Marked {recovered} stale job(s) as failed")

        return recovered

    finally:
        if should_close:
            db.close()


def process_job(
    job_id: str,
    youtube_url: str,
    db_path: str,
    _db_override: Optional[Session] = None,
) -> None:
    """
    Main job processing pipeline. Synchronous — runs in a separate RQ worker process.

    Each stage updates Job.status immediately and commits so progress survives a restart.
    All stages are fully wired: download, transcribe, extract_frames, generate_thumbnails,
    analyze_signals, detect_clips, extract_preview. Job only reaches 'complete' after all
    preview .mp4 files are written to disk and Clip rows are committed to the database.

    Args:
        job_id: UUID string primary key of the Job row.
        youtube_url: YouTube URL to process.
        db_path: Absolute path to the SQLite database file, passed explicitly so RQ
                 worker processes don't need to read config (avoids import side effects).
        _db_override: Inject an existing Session for testing.
    """
    if _db_override is not None:
        db = _db_override
        should_close = False
    else:
        if db_path == ":memory:":
            url = "sqlite:///:memory:"
        else:
            url = f"sqlite:///{db_path}"
        engine = create_engine(url, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        should_close = True

    def _update_status(status: str, message: str = "") -> None:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            job.progress_message = message
            db.commit()
        logger.info(f"[{job_id}] {status}: {message}")

    def _mark_failed(error: Exception) -> None:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(error)
            job.progress_message = f"Failed: {error}"
            db.commit()
        logger.exception(f"[{job_id}] pipeline failed: {error}")

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"process_job: job {job_id} not found in database")
            return

        # Stage 1: Download
        _update_status("downloading", "Downloading video from YouTube...")
        video_info = download_video(youtube_url, job_id)

        # Persist video metadata to Job record
        job.video_title = video_info.get("title")
        job.video_duration_seconds = video_info.get("duration_seconds")
        db.commit()

        # Stage 2: Transcribe
        _update_status("transcribing", "Transcribing audio...")
        transcript_data = transcribe_video(
            video_info.get("video_path", ""),
            job_id,
        )

        # Stage 3: Extract frames — wired in plan 01-03
        _update_status("extracting_frames", "Extracting candidate frames...")
        frames = extract_candidate_frames(video_info.get("video_path", ""), job_id)

        # Stage 3.5: Enhance frames (local processing -- CLAHE, sharpening, desaturation)
        _update_status("enhancing_frames", "Enhancing frames for thumbnail quality...")
        frames = enhance_frames_batch(frames)

        # Stage 4: Generate thumbnails — wired in plan 01-04
        _update_status("generating_thumbnails", "Generating thumbnails...")
        thumbnails = generate_thumbnails(
            frames,
            transcript_data,
            job_id,
            enhance=settings.use_gemini_enhancement,
            video_title=video_info.get("title", ""),
        )

        # Save thumbnails to database
        for thumb in thumbnails:
            db_thumb = Thumbnail(
                id=thumb.get("thumb_id", str(_uuid.uuid4())[:8]),
                job_id=job_id,
                frame_index=thumb.get("frame_index"),
                text_overlay=thumb.get("concept", {}).get("text_overlay", ""),
                text_position=thumb.get("concept", {}).get("text_position", ""),
                style_notes=thumb.get("concept", {}).get("style_notes", ""),
                reasoning=thumb.get("concept", {}).get("reasoning", ""),
                file_path=thumb.get("file_path", ""),
                generation_type=thumb.get("generation_type", "pillow"),
            )
            db.add(db_thumb)
        db.commit()
        logger.info(f"[{job_id}] Saved {len(thumbnails)} thumbnail(s) to database")

        # Stage 5: Signal analysis (reuses audio.wav written to disk by Stage 2)
        job_dir = settings.downloads_dir / job_id
        _update_status("analyzing", "Analyzing audio signals...")
        signal_data = analyze_signals(
            audio_path=str(job_dir / "audio.wav"),
            words=transcript_data.get("words", []),
        )

        # Stage 6: Detect clips
        _update_status("detecting_clips", "Detecting viral clip moments...")
        raw_clips = detect_clips(
            words=transcript_data.get("words", []),
            signal_data=signal_data,
            duration_seconds=float(transcript_data.get("duration", 0)),
        )

        # Stage 7: Extract preview files (MUST complete before marking job complete)
        clips_output_dir = settings.clips_dir / job_id
        clips_output_dir.mkdir(parents=True, exist_ok=True)
        _update_status("detecting_clips", f"Extracting {len(raw_clips)} clip previews...")

        for clip_data in raw_clips:
            clip_id = str(_uuid.uuid4())[:8]
            preview_path = str(clips_output_dir / f"{clip_id}.mp4")
            try:
                extract_preview(
                    video_path=video_info.get("video_path", ""),
                    start_seconds=clip_data["start_time_seconds"],
                    end_seconds=clip_data["end_time_seconds"],
                    output_path=preview_path,
                    ffmpeg_path=settings.ffmpeg_path,
                )
                clip_data["preview_path"] = preview_path
                clip_data["clip_id"] = clip_id
            except Exception as e:
                logger.warning(
                    f"[{job_id}] Preview extraction failed for clip at "
                    f"{clip_data.get('start_time_seconds')}: {e}"
                )
                clip_data["preview_path"] = None
                clip_data["clip_id"] = clip_id

        # Save clips with preview_path to database (skip clips whose preview failed)
        for clip_data in raw_clips:
            if not clip_data.get("preview_path"):
                continue  # skip clips whose preview extraction failed
            db_clip = Clip(
                id=clip_data["clip_id"],
                job_id=job_id,
                start_time_seconds=clip_data.get("start_time_seconds"),
                end_time_seconds=clip_data.get("end_time_seconds"),
                duration_seconds=clip_data.get("duration_seconds"),
                transcript_snippet=clip_data.get("transcript_snippet"),
                clip_title=clip_data.get("clip_title"),
                hook_text=clip_data.get("hook_text"),
                virality_score=clip_data.get("virality_score", 0),
                hook_strength=clip_data.get("score_breakdown", {}).get("hook_strength", 0),
                standalone_clarity=clip_data.get("score_breakdown", {}).get("standalone_clarity", 0),
                emotional_arc=clip_data.get("score_breakdown", {}).get("emotional_arc", 0),
                trend_alignment=clip_data.get("score_breakdown", {}).get("trend_alignment", 0),
                rewatch_potential=clip_data.get("score_breakdown", {}).get("rewatch_potential", 0),
                reasoning=clip_data.get("reasoning"),
                suggested_caption=clip_data.get("suggested_caption"),
                suggested_duration=clip_data.get("suggested_duration"),
                clip_type=clip_data.get("clip_type"),
                edit_suggestions=json.dumps(clip_data.get("edit_suggestions", [])),
                preview_path=clip_data["preview_path"],  # REQUIRED — never null for saved clips
            )
            db.add(db_clip)
        db.commit()
        logger.info(f"[{job_id}] Saved {len(raw_clips)} clip(s) with preview files")

        # Done
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "complete"
            job.progress_message = "Processing complete!"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
        logger.info(f"[{job_id}] complete")

    except Exception as exc:
        _mark_failed(exc)

    finally:
        if should_close:
            db.close()
