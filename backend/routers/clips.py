from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import Job, Clip, Thumbnail, get_db

router = APIRouter(prefix="/api/jobs/{job_id}/clips", tags=["clips"])


class ScoreBreakdown(BaseModel):
    hook_strength: int
    standalone_clarity: int
    emotional_arc: int
    trend_alignment: int
    rewatch_potential: int


class EditSuggestion(BaseModel):
    type: str
    suggestion: str
    reference: Optional[str] = None
    priority: str = "medium"


class ClipResponse(BaseModel):
    id: str
    job_id: str
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    transcript_snippet: Optional[str] = None
    clip_title: Optional[str] = None
    hook_text: Optional[str] = None
    virality_score: int
    score_breakdown: ScoreBreakdown
    reasoning: Optional[str] = None
    suggested_caption: Optional[str] = None
    suggested_duration: Optional[str] = None
    clip_type: Optional[str] = None
    preview_url: Optional[str] = None
    kept: bool = False
    edit_suggestions: List[EditSuggestion] = []

    class Config:
        from_attributes = True


class ClipUpdateRequest(BaseModel):
    kept: bool


@router.get("", response_model=List[ClipResponse])
def list_clips(job_id: str, db: Session = Depends(get_db)):
    """Get all detected clips for a job, sorted by virality score."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clips = (
        db.query(Clip)
        .filter(Clip.job_id == job_id)
        .order_by(Clip.virality_score.desc())
        .all()
    )

    result = []
    for c in clips:
        # Parse edit_suggestions from JSON string
        suggestions = []
        if c.edit_suggestions:
            try:
                raw = json.loads(c.edit_suggestions)
                suggestions = [EditSuggestion(**s) for s in raw if isinstance(s, dict)]
            except (json.JSONDecodeError, Exception):
                pass

        result.append(ClipResponse(
            id=c.id,
            job_id=c.job_id,
            start_time_seconds=c.start_time_seconds or 0,
            end_time_seconds=c.end_time_seconds or 0,
            duration_seconds=c.duration_seconds or 0,
            transcript_snippet=c.transcript_snippet,
            clip_title=c.clip_title,
            hook_text=c.hook_text,
            virality_score=c.virality_score or 0,
            score_breakdown=ScoreBreakdown(
                hook_strength=c.hook_strength or 0,
                standalone_clarity=c.standalone_clarity or 0,
                emotional_arc=c.emotional_arc or 0,
                trend_alignment=c.trend_alignment or 0,
                rewatch_potential=c.rewatch_potential or 0,
            ),
            reasoning=c.reasoning,
            suggested_caption=c.suggested_caption,
            suggested_duration=c.suggested_duration,
            clip_type=c.clip_type,
            preview_url=f"/api/jobs/{job_id}/clips/{c.id}/preview" if c.preview_path else None,
            kept=bool(c.kept),
            edit_suggestions=suggestions,
        ))
    return result


@router.get("/{clip_id}/preview")
def get_clip_preview(job_id: str, clip_id: str, db: Session = Depends(get_db)):
    """Serve the playable .mp4 preview for a clip."""
    clip = db.query(Clip).filter(Clip.id == clip_id, Clip.job_id == job_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.preview_path or not Path(clip.preview_path).exists():
        raise HTTPException(status_code=404, detail="Preview not available")
    return FileResponse(
        clip.preview_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},  # required for browser seeking
    )


@router.patch("/{clip_id}")
def update_clip(job_id: str, clip_id: str, body: ClipUpdateRequest, db: Session = Depends(get_db)):
    """Update the kept state of a clip."""
    clip = db.query(Clip).filter(Clip.id == clip_id, Clip.job_id == job_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip.kept = 1 if body.kept else 0
    db.commit()
    return {"id": clip_id, "kept": bool(clip.kept)}


@router.get("/{clip_id}/thumbnail")
def get_clip_thumbnail(job_id: str, clip_id: str, db: Session = Depends(get_db)):
    """Download the thumbnail associated with a clip's job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    thumbnails = db.query(Thumbnail).filter(Thumbnail.job_id == job_id).all()
    if not thumbnails:
        raise HTTPException(status_code=404, detail="No thumbnails available")
    # Return the first thumbnail (primary)
    thumb = thumbnails[0]
    file_path = Path(thumb.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=f"thumbnail_{job_id}.jpg",
        headers={"Content-Disposition": f'attachment; filename="thumbnail_{job_id}.jpg"'},
    )
