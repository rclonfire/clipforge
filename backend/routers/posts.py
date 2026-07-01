"""
Post-prep router — "Prep for posting" feature.

Turns a completed export batch into a ready-to-post bundle (per-clip copy,
thumbnail, and a posting schedule) for YouTube Shorts and TikTok. The user
downloads the bundle and posts via each platform's own scheduler.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Job, Clip, Export, ExportBatch, Thumbnail, get_db
from backend.services.post_prep import build_bundle, build_schedule, generate_post_copy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs/{job_id}/post-prep", tags=["post-prep"])

_ALLOWED_PLATFORMS = {"youtube", "tiktok"}


class PostPrepRequest(BaseModel):
    batch_id: str | None = None  # defaults to the job's latest complete export batch
    platforms: list[str] = ["youtube", "tiktok"]
    posts_per_day: int = 3
    start_date: str | None = None  # ISO date (YYYY-MM-DD); defaults to today


def _prep_dir(batch_id: str) -> Path:
    return settings.exports_dir / f"postready_{batch_id}"


@router.post("")
def create_post_prep(job_id: str, body: PostPrepRequest, db: Session = Depends(get_db)):
    """Generate platform copy + schedule + bundle for a completed export batch."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Resolve the export batch (explicit, or the job's latest complete one)
    if body.batch_id:
        batch = db.query(ExportBatch).filter(
            ExportBatch.id == body.batch_id, ExportBatch.job_id == job_id
        ).first()
    else:
        batch = (
            db.query(ExportBatch)
            .filter(ExportBatch.job_id == job_id, ExportBatch.status == "complete")
            .order_by(ExportBatch.created_at.desc())
            .first()
        )
    if not batch:
        raise HTTPException(status_code=404, detail="No export batch found — export clips first")
    if batch.status != "complete":
        raise HTTPException(status_code=400, detail="Export batch is not complete yet")

    platforms = [p for p in body.platforms if p in _ALLOWED_PLATFORMS]
    if not platforms:
        raise HTTPException(status_code=400, detail="No valid platforms (use 'youtube' / 'tiktok')")
    posts_per_day = max(1, min(3, body.posts_per_day))

    try:
        start_date = date.fromisoformat(body.start_date) if body.start_date else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="start_date must be YYYY-MM-DD")

    # Map exported clip files for this batch
    exports = db.query(Export).filter(Export.batch_id == batch.id).all()
    file_map = {e.clip_id: e.file_path for e in exports if e.file_path}
    if not file_map:
        raise HTTPException(status_code=404, detail="No exported clip files for this batch")

    # Kept clips that actually have an exported file, in playback order
    clips = (
        db.query(Clip)
        .filter(Clip.job_id == job_id, Clip.kept == 1)
        .order_by(Clip.start_time_seconds)
        .all()
    )
    clips = [c for c in clips if c.id in file_map]
    if not clips:
        raise HTTPException(status_code=400, detail="No kept, exported clips to prep")

    # Job-level thumbnail (used as the YouTube cover for long-form re-uploads)
    thumb = db.query(Thumbnail).filter(Thumbnail.job_id == job_id).first()
    thumb_path = thumb.file_path if thumb else None

    song = {
        "song": job.song_title or "",
        "artist": job.song_artist or "",
        "confidence": job.song_confidence or "none",
    }
    copy_map = generate_post_copy(clips, job.video_title, song=song)
    schedule = build_schedule([c.id for c in clips], platforms, posts_per_day, start_date)

    prep_dir = _prep_dir(batch.id)
    build_bundle(prep_dir, clips, copy_map, file_map, thumb_path, schedule, job.video_title)
    logger.info("[post-prep:%s] Built bundle for %d clip(s), platforms=%s", batch.id, len(clips), platforms)

    return {
        "prep_id": batch.id,
        "batch_id": batch.id,
        "platforms": platforms,
        "posts_per_day": posts_per_day,
        "start_date": start_date.isoformat(),
        "clip_count": len(clips),
        "song": {"title": song["song"], "artist": song["artist"], "confidence": song["confidence"]},
        "schedule": schedule,
        "clips": [
            {
                "clip_index": i,
                "clip_id": c.id,
                "clip_title": c.clip_title,
                "youtube": copy_map[c.id]["youtube"],
                "tiktok": copy_map[c.id]["tiktok"],
            }
            for i, c in enumerate(clips, start=1)
        ],
        "download_url": f"/api/jobs/{job_id}/post-prep/{batch.id}/download",
    }


@router.get("/{prep_id}/download")
def download_post_prep(job_id: str, prep_id: str, db: Session = Depends(get_db)):
    """Download the post-ready bundle as a zip."""
    batch = db.query(ExportBatch).filter(
        ExportBatch.id == prep_id, ExportBatch.job_id == job_id
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Prep batch not found")

    prep_dir = _prep_dir(prep_id)
    if not prep_dir.exists():
        raise HTTPException(status_code=404, detail="No prepped bundle — run post-prep first")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(prep_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(prep_dir)))
    buf.seek(0)

    job = db.query(Job).filter(Job.id == job_id).first()
    name = "clipforge_postready"
    if job and job.video_title:
        safe = "".join(c for c in job.video_title if c.isalnum() or c in " -_")[:50].strip()
        if safe:
            name = f"{safe}_postready"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )
