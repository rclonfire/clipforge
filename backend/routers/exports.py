from __future__ import annotations

import io
import logging
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import Job, Clip, Export, ExportBatch, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs/{job_id}/exports", tags=["exports"])


class ExportRequest(BaseModel):
    platform: str = "tiktok"  # "tiktok" | "shorts" | "original"
    vertical_crop: bool = True
    captions: bool = True


class ExportStatusResponse(BaseModel):
    batch_id: str
    status: str
    progress_message: str
    total_clips: int
    completed_clips: int


@router.post("")
def create_export(job_id: str, body: ExportRequest, db: Session = Depends(get_db)):
    """Create an export batch job for all kept clips."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    kept_clips = db.query(Clip).filter(Clip.job_id == job_id, Clip.kept == 1).all()
    if not kept_clips:
        raise HTTPException(status_code=400, detail="No kept clips to export")

    batch_id = str(uuid.uuid4())[:12]
    batch = ExportBatch(
        id=batch_id,
        job_id=job_id,
        platform=body.platform,
        vertical_crop=1 if body.vertical_crop else 0,
        captions=1 if body.captions else 0,
        status="pending",
        total_clips=len(kept_clips),
        completed_clips=0,
        progress_message="Export queued...",
    )
    db.add(batch)
    db.commit()

    # Enqueue export job
    from backend.config import settings as _settings
    _enqueue_export(batch_id, job_id, str(_settings.db_path))

    return {"batch_id": batch_id, "status": "pending", "total_clips": len(kept_clips)}


@router.get("/{batch_id}/status", response_model=ExportStatusResponse)
def export_status(job_id: str, batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(ExportBatch).filter(
        ExportBatch.id == batch_id, ExportBatch.job_id == job_id
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Export batch not found")
    return ExportStatusResponse(
        batch_id=batch.id,
        status=batch.status,
        progress_message=batch.progress_message or "",
        total_clips=batch.total_clips or 0,
        completed_clips=batch.completed_clips or 0,
    )


@router.get("/{batch_id}/download")
def download_export_zip(job_id: str, batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(ExportBatch).filter(
        ExportBatch.id == batch_id, ExportBatch.job_id == job_id
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Export batch not found")
    if batch.status != "complete":
        raise HTTPException(status_code=400, detail="Export not yet complete")

    exports = db.query(Export).filter(Export.batch_id == batch_id).all()
    if not exports:
        raise HTTPException(status_code=404, detail="No export files found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for exp in exports:
            if exp.file_path and Path(exp.file_path).exists():
                zf.write(exp.file_path, arcname=Path(exp.file_path).name)
            # Include thumbnail if the clip has one (from job thumbnails)
            clip = db.query(Clip).filter(Clip.id == exp.clip_id).first()
            if clip:
                from backend.models import Thumbnail
                thumb = db.query(Thumbnail).filter(
                    Thumbnail.job_id == clip.job_id
                ).first()
                if thumb and thumb.file_path and Path(thumb.file_path).exists():
                    thumb_name = f"thumbnail_{clip.id}.jpg"
                    # Avoid duplicate entries in zip
                    if thumb_name not in [info.filename for info in zf.infolist()]:
                        zf.write(thumb.file_path, arcname=thumb_name)
    buf.seek(0)

    video_title = "clipforge_export"
    job = db.query(Job).filter(Job.id == job_id).first()
    if job and job.video_title:
        # Sanitize title for filename
        safe_title = "".join(c for c in job.video_title if c.isalnum() or c in " -_")[:50].strip()
        if safe_title:
            video_title = safe_title

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{video_title}.zip"'},
    )


def _enqueue_export(batch_id: str, job_id: str, db_path: str) -> None:
    """Enqueue export processing via RQ or thread fallback."""
    from backend.config import settings

    try:
        import redis as redis_lib
        from rq import Queue

        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()

        from backend.workers.export_processor import process_export_batch
        q = Queue(connection=r)
        q.enqueue(
            process_export_batch,
            batch_id,
            job_id,
            db_path,
            result_ttl=86400,
            failure_ttl=604800,
            job_timeout=1800,  # 30 min timeout for batch export
        )
        logger.info(f"[export:{batch_id}] Enqueued via RQ")

    except Exception as exc:
        logger.warning(f"[export:{batch_id}] Redis unavailable ({exc}) — thread fallback")
        import threading
        from backend.models import SessionLocal

        def _run():
            from backend.workers.export_processor import process_export_batch
            process_export_batch(batch_id, job_id, db_path)

        thread = threading.Thread(target=_run, daemon=True, name=f"export-{batch_id}")
        thread.start()
