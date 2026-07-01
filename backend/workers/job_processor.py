"""
RQ-compatible job processor for ClipForge.

IMPORTANT: process_job() / process_local_job() are regular (sync) functions —
NOT async. RQ workers run in separate processes outside the FastAPI event loop.
Each worker call opens its own SQLite session from db_path.

Two ingest entrypoints share the same downstream pipeline:
    process_job(job_id, youtube_url, db_path)        — Stage 1 = yt-dlp download
    process_local_job(job_id, source_path, db_path)  — Stage 1 = import a local file

Pipeline stages (Stages 2-7 live in _run_pipeline, shared by both entrypoints):
    1. obtain video        — download_video OR ingest_local_file
    2. transcribe_video
    3. extract_frames
    4. generate_thumbnails
    5. analyze_signals     — uses audio.wav from Stage 2
    6. detect_clips        — Claude clip detection
    7. extract_preview     — FFmpeg .mp4 preview per clip

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
from backend.services.video_ingestion import download_video, ingest_local_file
from backend.services.transcription import transcribe_video
from backend.services.frame_extraction import extract_candidate_frames
from backend.services.frame_enhancer import enhance_frames_batch
from backend.services.thumbnail_generator import generate_thumbnails
from backend.services.signal_analysis import analyze_signals
from backend.services.clip_detector import detect_clips
from backend.services.music_clip_detector import detect_music_clips, looks_instrumental
from backend.services.preview_extractor import extract_preview

logger = logging.getLogger(__name__)

# Transient statuses eligible for stale job recovery
_STALE_STATUSES = (
    "ingesting",
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _open_session(db_path: str) -> Session:
    """Open a standalone SQLite session for a worker process."""
    url = "sqlite:///:memory:" if db_path == ":memory:" else f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    return sessionmaker(bind=engine)()


def _set_status(db: Session, job_id: str, status: str, message: str = "") -> None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.progress_message = message
        db.commit()
    logger.info(f"[{job_id}] {status}: {message}")


def _mark_failed(db: Session, job_id: str, error: Exception) -> None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = "failed"
        job.error_message = str(error)
        job.progress_message = f"Failed: {error}"
        db.commit()
    logger.exception(f"[{job_id}] pipeline failed: {error}")


# ---------------------------------------------------------------------------
# Ingest entrypoints
# ---------------------------------------------------------------------------

def process_job(
    job_id: str,
    youtube_url: str,
    db_path: str,
    _db_override: Optional[Session] = None,
) -> None:
    """
    Process a YouTube job: download via yt-dlp, then run the shared pipeline.

    Synchronous — runs in a separate RQ worker process. Each stage updates
    Job.status and commits so progress survives a restart.

    Args:
        job_id: UUID string primary key of the Job row.
        youtube_url: YouTube URL to process.
        db_path: Absolute path to the SQLite database file (or ':memory:' for tests).
        _db_override: Inject an existing Session for testing.
    """
    if _db_override is not None:
        db, should_close = _db_override, False
    else:
        db, should_close = _open_session(db_path), True

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"process_job: job {job_id} not found in database")
            return

        _set_status(db, job_id, "downloading", "Downloading video from YouTube...")
        video_info = download_video(youtube_url, job_id)
        _run_pipeline(db, job_id, video_info)

    except Exception as exc:
        _mark_failed(db, job_id, exc)
    finally:
        if should_close:
            db.close()


def process_local_job(
    job_id: str,
    source_path: str,
    db_path: str,
    _db_override: Optional[Session] = None,
) -> None:
    """
    Process a locally-supplied video file: import it, then run the shared pipeline.

    Same shape as process_job but Stage 1 imports a file from disk instead of
    downloading from YouTube. Used for editor exports that never touch YouTube.

    Args:
        job_id: UUID string primary key of the Job row.
        source_path: Absolute path to the local video file.
        db_path: Absolute path to the SQLite database file (or ':memory:' for tests).
        _db_override: Inject an existing Session for testing.
    """
    if _db_override is not None:
        db, should_close = _db_override, False
    else:
        db, should_close = _open_session(db_path), True

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"process_local_job: job {job_id} not found in database")
            return

        _set_status(db, job_id, "ingesting", "Importing local video file...")
        video_info = ingest_local_file(source_path, job_id)
        _run_pipeline(db, job_id, video_info)

    except Exception as exc:
        _mark_failed(db, job_id, exc)
    finally:
        if should_close:
            db.close()


# ---------------------------------------------------------------------------
# Shared pipeline (Stages 2-7)
# ---------------------------------------------------------------------------

def _run_pipeline(db: Session, job_id: str, video_info: dict) -> None:
    """
    Run Stages 2-7 on an obtained video. Source-agnostic: video_info provides the
    local .mp4 path plus title/duration, whether it came from YouTube or a local file.

    Job only reaches 'complete' after all preview .mp4 files are written and Clip
    rows are committed.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    job.video_title = video_info.get("title")
    job.video_duration_seconds = video_info.get("duration_seconds")
    db.commit()

    video_path = video_info.get("video_path", "")

    # Stage 2: Transcribe
    _set_status(db, job_id, "transcribing", "Transcribing audio...")
    transcript_data = transcribe_video(video_path, job_id)

    # Stage 2.5: Identify the song (autonomous, for caption naming). Best-effort —
    # instrumental covers can't be transcribed/fingerprinted, so a Gemini audio model
    # names the melody when it can; failures degrade to vibe-only captions.
    try:
        from backend.services.song_identify import identify_song
        _set_status(db, job_id, "transcribing", "Identifying song...")
        song = identify_song(
            str(settings.downloads_dir / job_id / "audio.wav"),
            ffmpeg_path=settings.ffmpeg_path,
        )
        song_job = db.query(Job).filter(Job.id == job_id).first()
        song_job.song_title = song.get("song") or None
        song_job.song_artist = song.get("artist") or None
        song_job.song_confidence = song.get("confidence") or "none"
        db.commit()
    except Exception as exc:  # noqa: BLE001 — never block the pipeline
        logger.warning("[%s] song identification skipped: %s", job_id, exc)

    # Stage 3: Extract frames
    _set_status(db, job_id, "extracting_frames", "Extracting candidate frames...")
    frames = extract_candidate_frames(video_path, job_id)

    # Stage 3.5: Enhance frames (local processing -- CLAHE, sharpening, desaturation)
    _set_status(db, job_id, "enhancing_frames", "Enhancing frames for thumbnail quality...")
    frames = enhance_frames_batch(frames)

    # Stage 4: Generate thumbnails — paid (Claude concepts + optional Gemini polish).
    # Skipped entirely in free mode; Shorts/TikTok pick a cover frame themselves.
    if settings.use_paid_apis:
        _set_status(db, job_id, "generating_thumbnails", "Generating thumbnails...")
        thumbnails = generate_thumbnails(
            frames,
            transcript_data,
            job_id,
            enhance=settings.use_gemini_enhancement,
            video_title=video_info.get("title", ""),
        )

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
    else:
        logger.info(f"[{job_id}] Free mode: skipping AI thumbnail generation (no paid calls)")

    # Stage 5: Signal analysis (reuses audio.wav written to disk by Stage 2)
    job_dir = settings.downloads_dir / job_id
    _set_status(db, job_id, "analyzing", "Analyzing audio signals...")
    signal_data = analyze_signals(
        audio_path=str(job_dir / "audio.wav"),
        words=transcript_data.get("words", []),
    )

    # Stage 6: Detect clips.
    # - Free mode OR instrumental content -> music/energy brain (librosa energy
    #   fallback is fully local; Gemini is used only when paid APIs are enabled).
    # - Paid mode + speech content -> the Claude comedy/transcript brain.
    words = transcript_data.get("words", [])
    duration = float(transcript_data.get("duration", 0))
    if settings.use_paid_apis and not looks_instrumental(words, duration):
        _set_status(db, job_id, "detecting_clips", "Detecting viral clip moments...")
        raw_clips = detect_clips(
            words=words,
            signal_data=signal_data,
            duration_seconds=duration,
        )
    else:
        _set_status(db, job_id, "detecting_clips", "Detecting best moments...")
        raw_clips = detect_music_clips(
            audio_path=str(job_dir / "audio.wav"),
            signal_data=signal_data,
            duration_seconds=duration,
            ffmpeg_path=settings.ffmpeg_path,
        )

    # Stage 7: Extract preview files (MUST complete before marking job complete)
    clips_output_dir = settings.clips_dir / job_id
    clips_output_dir.mkdir(parents=True, exist_ok=True)
    _set_status(db, job_id, "detecting_clips", f"Extracting {len(raw_clips)} clip previews...")

    for clip_data in raw_clips:
        clip_id = str(_uuid.uuid4())[:8]
        preview_path = str(clips_output_dir / f"{clip_id}.mp4")
        try:
            extract_preview(
                video_path=video_path,
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
            continue
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
            preview_path=clip_data["preview_path"],
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
