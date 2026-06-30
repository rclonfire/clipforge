"""
Export encoder service for ClipForge — Phase 03-export-engine.

Produces final export-quality MP4 files using FFmpeg with:
- CRF 18, H.264 High profile for high quality
- 9:16 vertical crop with face-center detection (tiktok/shorts)
- 16:9 scale-only output (original preset)
- Optional ASS caption burn-in
- Platform duration clamping (60s max for tiktok/shorts)

Interface:
    export_clip(video_path, start_seconds, end_seconds, output_path,
                ass_path=None, vertical_crop=True, platform="tiktok",
                ffmpeg_path=...) -> str

Key behaviors:
- Uses exact timestamps (no pre/post-roll — that's for preview only)
- Duration clamped to platform max_duration
- Face detected at start_seconds + 1.0 for crop centering
- _detect_face_center and _get_video_dimensions reused from preview_extractor.py
- subprocess.run with check=True, capture_output=True, timeout=600
"""
from __future__ import annotations

import logging
import subprocess

from backend.services.preview_extractor import (
    _detect_face_center,
    _get_video_dimensions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------
PLATFORM_PRESETS: dict[str, dict] = {
    "tiktok": {
        "max_duration": 60,
        "width": 1080,
        "height": 1920,
        "vertical": True,
    },
    "shorts": {
        "max_duration": 60,
        "width": 1080,
        "height": 1920,
        "vertical": True,
    },
    "original": {
        "max_duration": 300,
        "width": 1920,
        "height": 1080,
        "vertical": False,
    },
}


def _build_vf_filter(
    crop_w: int,
    crop_h: int,
    crop_x: int,
    crop_y: int,
    out_w: int,
    out_h: int,
    vertical_crop: bool,
    ass_path: str | None = None,
) -> str:
    """
    Build the FFmpeg -vf filter string for crop/scale and optional caption burn-in.

    Args:
        crop_w, crop_h: Crop region dimensions (ignored when vertical_crop=False).
        crop_x, crop_y: Crop region origin (ignored when vertical_crop=False).
        out_w, out_h: Output (scaled) dimensions.
        vertical_crop: When True, prepend a crop filter.
        ass_path: Path to .ass file for caption burn-in (optional).

    Returns:
        Complete -vf filter string, e.g. "crop=608:1080:656:0,scale=1080:1920,ass=/tmp/x.ass"
    """
    if vertical_crop:
        vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h}"
    else:
        vf = f"scale={out_w}:{out_h}"

    if ass_path:
        escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:")
        vf = f"{vf},ass={escaped}"

    return vf


def export_clip(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    ass_path: str | None = None,
    vertical_crop: bool = True,
    platform: str = "tiktok",
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg",
) -> str:
    """
    Encode and export a clip to MP4 with optional vertical crop and caption burn-in.

    Uses exact start/end timestamps (no pre/post-roll — that is applied only
    for preview extraction in preview_extractor.py).

    Args:
        video_path: Absolute path to the source video.
        start_seconds: Clip start time in seconds.
        end_seconds: Clip end time in seconds.
        output_path: Absolute path for the output .mp4 file.
        ass_path: Optional path to .ass subtitle file for caption burn-in.
        vertical_crop: When True, crops to 9:16 vertical with face centering.
                       When False, scales to 16:9 (original preset).
        platform: One of "tiktok", "shorts", "original".
        ffmpeg_path: Path to the ffmpeg binary.

    Returns:
        output_path — same string passed in, on success.
    """
    preset = PLATFORM_PRESETS[platform]

    # Calculate and clamp duration
    duration = end_seconds - start_seconds
    duration = min(duration, preset["max_duration"])

    # Default crop/scale params (overridden below if vertical_crop is True)
    crop_w = crop_h = crop_x = crop_y = 0
    out_w = preset["width"]
    out_h = preset["height"]

    if vertical_crop:
        # Get source dimensions
        try:
            src_w, src_h = _get_video_dimensions(video_path, ffmpeg_path)
        except Exception:
            src_w, src_h = 1920, 1080  # fallback assumption

        # 9:16 target ratio (width/height)
        target_ratio = 9 / 16
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            # Source is wider — crop width, keep full height
            crop_h = src_h
            crop_w = int(src_h * target_ratio)
        else:
            # Source is taller — crop height, keep full width
            crop_w = src_w
            crop_h = int(src_w / target_ratio)

        # Detect face to center crop
        face_center = _detect_face_center(video_path, start_seconds + 1.0, ffmpeg_path)

        if face_center:
            face_x_frac, face_y_frac = face_center
            crop_x = int(face_x_frac * src_w - crop_w / 2)
            crop_y = int(face_y_frac * src_h - crop_h / 2)
        else:
            crop_x = (src_w - crop_w) // 2
            crop_y = (src_h - crop_h) // 2

        # Clamp to valid range
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        # Ensure even output dimensions for H.264
        out_w = preset["width"] - (preset["width"] % 2)
        out_h = preset["height"] - (preset["height"] % 2)

    # Build video filter
    vf_str = _build_vf_filter(
        crop_w, crop_h, crop_x, crop_y,
        out_w, out_h,
        vertical_crop,
        ass_path,
    )

    # Build FFmpeg command
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss", str(start_seconds),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf_str,
        "-c:v", "libx264",
        "-profile:v", "high",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info(
        "Exporting clip: %.1fs-%.1fs (%.1fs) platform=%s vertical_crop=%s ass=%s -> %s",
        start_seconds, end_seconds, duration, platform, vertical_crop,
        ass_path or "none", output_path,
    )

    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    return output_path
