"""
RQ-compatible export processor for ClipForge.

Processes a batch export: for each kept clip, optionally runs WhisperX caption
alignment, generates ASS subtitle file, then encodes via FFmpeg.

Pipeline per clip:
    1. (optional) align_words_for_clip -> word-level timestamps
    2. (optional) generate_ass -> .ass subtitle file
    3. export_clip -> .mp4 with crop + captions

Same DB session pattern as job_processor.py: opens own SQLite session from db_path,
or accepts an injected session via _db_override for test isolation.
"""
from __future__ import annotations

import json
import logging
import os
import uuid as _uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.models import Base, Job, Clip, Export, ExportBatch
from backend.config import settings
from backend.services.caption_aligner import align_words_for_clip
from backend.services.ass_generator import generate_ass
from backend.services.export_encoder import export_clip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_title(title: str) -> str:
    """
    Return a filesystem-safe string from a video title.

    Keeps alphanumeric characters, spaces, hyphens, and underscores.
    Truncates to 50 characters, strips leading/trailing whitespace, and
    replaces spaces with underscores.

    Returns "clip" if the result would be empty.
    """
    sanitized = "".join(c for c in title if c.isalnum() or c in " -_")
    sanitized = sanitized[:50].strip().replace(" ", "_")
    return sanitized or "clip"


def _find_video_file(job_dir: Path) -> str:
    """
    Find the first video file (.mp4, .webm, .mkv) in job_dir.

    Args:
        job_dir: Directory to search for video files.

    Returns:
        Absolute path string to the first video file found.

    Raises:
        FileNotFoundError: If no video file is found in job_dir.
    """
    for ext in ("*.mp4", "*.webm", "*.mkv"):
        matches = list(job_dir.glob(ext))
        if matches:
            return str(matches[0].resolve())
    raise FileNotFoundError(f"No video file found in {job_dir}")


def _load_transcript(job_dir: Path) -> list[dict]:
    """
    Load word list from transcript.json in the job directory.

    Returns an empty list if the file doesn't exist or is malformed.
    """
    transcript_path = job_dir / "transcript.json"
    if not transcript_path.exists():
        logger.warning("transcript.json not found at %s — captions will be skipped", transcript_path)
        return []
    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        return data.get("words", [])
    except Exception as exc:
        logger.warning("Failed to load transcript.json: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------

def process_export_batch(
    batch_id: str,
    job_id: str,
    db_path: str,
    _db_override: Optional[Session] = None,
) -> None:
    """
    Process an export batch: encode each kept clip through the full pipeline.

    For each kept clip (ordered by start_time_seconds):
        1. (if captions) align_words_for_clip -> word-level timestamps
        2. (if captions) generate_ass -> subtitle file
        3. export_clip -> final .mp4 with optional vertical crop and captions

    Status transitions: pending -> processing -> complete (or failed).
    Per-clip failures are caught and logged — the batch continues processing
    remaining clips. Only a catastrophic top-level error sets status to "failed".

    Args:
        batch_id: ID of the ExportBatch record.
        job_id: ID of the parent Job record.
        db_path: Absolute path to the SQLite database file (or ":memory:" for tests).
                 Ignored when _db_override is provided.
        _db_override: Inject an existing Session for test isolation.
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
        # Load batch and job records
        batch = db.query(ExportBatch).filter(ExportBatch.id == batch_id).first()
        if not batch:
            logger.error("[export:%s] ExportBatch record not found", batch_id)
            return

        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error("[export:%s] Job %s not found", batch_id, job_id)
            batch.status = "failed"
            batch.error_message = f"Job {job_id} not found"
            db.commit()
            return

        # Transition to processing
        batch.status = "processing"
        batch.progress_message = "Starting export..."
        db.commit()
        logger.info("[export:%s] Starting batch export for job %s platform=%s", batch_id, job_id, batch.platform)

        # Locate source files
        job_dir = settings.downloads_dir / job_id
        try:
            video_path = _find_video_file(job_dir)
        except FileNotFoundError as exc:
            logger.error("[export:%s] %s", batch_id, exc)
            batch.status = "failed"
            batch.error_message = str(exc)
            db.commit()
            return

        audio_path = str(job_dir / "audio.wav")

        # Load transcript words (for caption alignment)
        words = _load_transcript(job_dir)

        # Create output directory for this batch
        export_dir = settings.exports_dir / batch_id
        export_dir.mkdir(parents=True, exist_ok=True)

        # Query all kept clips ordered by start time
        kept_clips = (
            db.query(Clip)
            .filter(Clip.job_id == job_id, Clip.kept == 1)
            .order_by(Clip.start_time_seconds)
            .all()
        )
        total = len(kept_clips)

        # Build safe video title for filenames
        safe_title = _sanitize_title(job.video_title or "clip")
        platform = batch.platform

        # Process each clip
        for clip_number, clip in enumerate(kept_clips, start=1):
            batch.progress_message = f"Exporting clip {clip_number}/{total}..."
            db.commit()

            try:
                # Build output filename: {safe_title}_{clip_number}_{platform}.mp4
                filename = f"{safe_title}_{clip_number}_{platform}.mp4"
                output_path = str(export_dir / filename)

                # Step 1+2: Caption alignment and ASS file generation (optional)
                ass_path: Optional[str] = None
                if batch.captions == 1:
                    aligned_words = align_words_for_clip(
                        audio_path=audio_path,
                        words=words,
                        clip_start=clip.start_time_seconds,
                        clip_end=clip.end_time_seconds,
                    )
                    ass_path = str(export_dir / f"clip_{clip_number}.ass")
                    generate_ass(aligned_words, ass_path)

                # Step 3: FFmpeg encode
                export_clip(
                    video_path=video_path,
                    start_seconds=clip.start_time_seconds,
                    end_seconds=clip.end_time_seconds,
                    output_path=output_path,
                    ass_path=ass_path,
                    vertical_crop=bool(batch.vertical_crop),
                    platform=platform,
                    ffmpeg_path=settings.ffmpeg_path,
                )

                # Record file size
                file_size = os.path.getsize(output_path)

                # Create Export row
                export_row = Export(
                    id=str(_uuid.uuid4())[:8],
                    clip_id=clip.id,
                    batch_id=batch_id,
                    platform=platform,
                    file_path=output_path,
                    file_size_bytes=file_size,
                )
                db.add(export_row)

                # Increment progress counter
                batch.completed_clips = clip_number
                db.commit()

                logger.info(
                    "[export:%s] Clip %d/%d done -> %s (%.1f KB)",
                    batch_id, clip_number, total, filename, file_size / 1024,
                )

            except Exception as clip_exc:
                logger.warning(
                    "[export:%s] Clip %d/%d failed: %s — continuing with remaining clips",
                    batch_id, clip_number, total, clip_exc,
                )
                # Do NOT re-raise — per-clip failure is isolated

        # Mark batch complete
        batch.status = "complete"
        batch.progress_message = f"Export complete — {batch.completed_clips} clip(s) ready"
        db.commit()
        logger.info("[export:%s] Batch complete — %d exports created", batch_id, batch.completed_clips)

    except Exception as exc:
        logger.exception("[export:%s] Unrecoverable batch error: %s", batch_id, exc)
        try:
            batch = db.query(ExportBatch).filter(ExportBatch.id == batch_id).first()
            if batch:
                batch.status = "failed"
                batch.error_message = str(exc)
                db.commit()
        except Exception:
            pass

    finally:
        if should_close:
            db.close()
