"""
Preview extractor service for ClipForge — Phase 02-03 (updated for vertical crops).

Extracts short .mp4 clip previews from a video file using FFmpeg.
Produces 9:16 vertical (1080x1920) previews with face-centered cropping.

Interface:
    extract_preview(video_path, start_seconds, end_seconds, output_path, ffmpeg_path) -> str

Key behaviors:
- -ss placed BEFORE -i (input seek) for fast seeking
- Vertical 9:16 crop centered on face position (detected via first frame)
- H.264 re-encode with CRF 23 for quality + small file size
- -movflags +faststart places moov atom at front for browser streaming
- Pre-roll: max(0.0, start_seconds - 1.0) — clamped at 0 for clips near start
- Post-roll: end_seconds + 0.5
- subprocess.run called with check=True, capture_output=True, timeout=300
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _detect_face_center(video_path: str, timestamp: float, ffmpeg_path: str) -> tuple[float, float] | None:
    """
    Extract a single frame at the given timestamp and detect the face center.
    Returns (x_fraction, y_fraction) relative to frame dimensions, or None if no face found.
    """
    try:
        import cv2
        import numpy as np

        # Extract a single frame using FFmpeg
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or not result.stdout:
            return None

        # Decode frame
        nparr = np.frombuffer(result.stdout, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Try MediaPipe face detection
        try:
            from backend.services.face_detection import detect_faces
            faces = detect_faces(frame)
            if faces:
                # Use the largest face (by area)
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                cx = (best.bbox[0] + best.bbox[2]) / 2 / w
                cy = (best.bbox[1] + best.bbox[3]) / 2 / h
                return (cx, cy)
        except Exception:
            pass

        # Fallback: OpenCV Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_cv = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        if len(faces_cv) > 0:
            # Largest face
            best = max(faces_cv, key=lambda f: f[2] * f[3])
            cx = (best[0] + best[2] / 2) / w
            cy = (best[1] + best[3] / 2) / h
            return (cx, cy)

        return None
    except Exception as e:
        logger.debug(f"Face detection failed for vertical crop: {e}")
        return None


def _get_video_dimensions(video_path: str, ffmpeg_path: str) -> tuple[int, int]:
    """Get video width and height using ffprobe."""
    ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_preview(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    output_path: str,
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg",
) -> str:
    """
    Extract a vertical 9:16 clip preview .mp4 with face-centered cropping.

    Detects face position from the first frame, then crops the horizontal video
    to a vertical window centered on the face. Re-encodes with H.264 CRF 23.

    Args:
        video_path: Absolute path to the source video file.
        start_seconds: Clip start time in seconds (before pre-roll is applied).
        end_seconds: Clip end time in seconds (before post-roll is applied).
        output_path: Absolute path where the output .mp4 should be written.
        ffmpeg_path: Path to the ffmpeg binary.

    Returns:
        output_path — the same string passed in, on success.
    """
    # Apply pre-roll and post-roll
    adj_start = max(0.0, start_seconds - 1.0)
    adj_end = end_seconds + 0.5
    duration = adj_end - adj_start

    # Get source video dimensions
    try:
        src_w, src_h = _get_video_dimensions(video_path, ffmpeg_path)
    except Exception:
        src_w, src_h = 1920, 1080  # fallback assumption

    # Target: 9:16 aspect ratio
    # Calculate crop dimensions from source
    target_ratio = 9 / 16  # width/height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider than target — crop width, keep full height
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        # Source is taller — crop height, keep full width
        crop_w = src_w
        crop_h = int(src_w / target_ratio)

    # Detect face to center the crop
    face_center = _detect_face_center(video_path, adj_start + 1.0, ffmpeg_path)

    if face_center:
        face_x_frac, face_y_frac = face_center
        # Center crop on face
        crop_x = int(face_x_frac * src_w - crop_w / 2)
        crop_y = int(face_y_frac * src_h - crop_h / 2)
    else:
        # No face found — center crop on frame
        crop_x = (src_w - crop_w) // 2
        crop_y = (src_h - crop_h) // 2

    # Clamp to valid range
    crop_x = max(0, min(crop_x, src_w - crop_w))
    crop_y = max(0, min(crop_y, src_h - crop_h))

    # Output resolution: 1080x1920 (or proportional if source is smaller)
    out_w = min(1080, crop_w)
    out_h = min(1920, crop_h)
    # Ensure even dimensions for H.264
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    # Build crop + scale filter
    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h}"

    logger.info(
        "Extracting vertical preview: %.1fs to %.1fs, crop=%dx%d@(%d,%d), face=%s -> %s",
        adj_start, adj_end, crop_w, crop_h, crop_x, crop_y,
        f"({face_center[0]:.2f},{face_center[1]:.2f})" if face_center else "center",
        output_path,
    )

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss", str(adj_start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return output_path
