from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import Job, Thumbnail, get_db

router = APIRouter(prefix="/api/jobs/{job_id}/thumbnails", tags=["thumbnails"])


class ThumbnailResponse(BaseModel):
    id: str
    job_id: str
    frame_index: Optional[int] = None
    text_overlay: Optional[str] = None
    text_position: Optional[str] = None
    style_notes: Optional[str] = None
    reasoning: Optional[str] = None
    estimated_ctr_tier: Optional[str] = None
    generation_type: Optional[str] = "gemini"
    file_url: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[ThumbnailResponse])
def list_thumbnails(job_id: str, db: Session = Depends(get_db)):
    """Get all generated thumbnails for a job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    thumbnails = db.query(Thumbnail).filter(Thumbnail.job_id == job_id).all()
    return [
        ThumbnailResponse(
            id=t.id,
            job_id=t.job_id,
            frame_index=t.frame_index,
            text_overlay=t.text_overlay,
            text_position=t.text_position,
            style_notes=t.style_notes,
            reasoning=t.reasoning,
            estimated_ctr_tier=t.estimated_ctr_tier,
            generation_type=t.generation_type or "gemini",
            file_url=f"/api/jobs/{job_id}/thumbnails/{t.id}/download",
        )
        for t in thumbnails
    ]


@router.get("/{thumbnail_id}/download")
def download_thumbnail(job_id: str, thumbnail_id: str, db: Session = Depends(get_db)):
    """Download a specific thumbnail image."""
    thumbnail = (
        db.query(Thumbnail)
        .filter(Thumbnail.id == thumbnail_id, Thumbnail.job_id == job_id)
        .first()
    )
    if not thumbnail:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    file_path = Path(thumbnail.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=f"thumbnail_{thumbnail_id}.jpg",
    )
