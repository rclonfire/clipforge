from __future__ import annotations

import subprocess
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from backend.config import FRAMES_DIR, MAX_CANDIDATE_FRAMES, FFMPEG_PATH, FFPROBE_PATH
from backend.services.face_detection import detect_faces, score_frame_quality

logger = logging.getLogger(__name__)


def extract_candidate_frames(
    video_path: str,
    job_id: str,
    max_frames: int = 40,
) -> list[dict]:
    """
    Extract candidate frames from ENTIRE video for thumbnail generation.

    CRITICAL: Ensures frames are distributed across the FULL VIDEO duration,
    not clustered in the beginning.

    Strategy:
    1. Divide video into time segments (ensures full coverage)
    2. Extract best frames from each segment using:
       - Scene change detection within segment
       - Visual quality scoring
       - Face detection
    3. Score and rank frames by composite (face_score * 0.7 + quality_score * 0.3)
    4. Resize selected frames to 512x288 (Claude-ready) after scoring
    5. Return top-scored frames distributed across the video

    Returns list of dicts with keys:
        frame_index, file_path, timestamp, face_score (0-1), quality_score (0-1)
    Additional keys (combined_score, has_face, etc.) may also be present.
    """
    output_dir = FRAMES_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration
    probe_cmd = [
        FFPROBE_PATH, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    duration = float(json.loads(result.stdout)["format"]["duration"])

    # STRATEGY: Extract frames distributed across ENTIRE video
    # Divide video into segments to ensure coverage
    num_segments = min(20, max(8, int(duration / 60)))  # 1 segment per ~1 minute for better coverage
    segment_duration = duration / num_segments

    logger.info(f"Dividing {duration:.0f}s video into {num_segments} segments for full coverage...")

    all_frames = []

    # Extract frames from each segment
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_duration
        seg_end = min((seg_idx + 1) * segment_duration, duration)

        # Try scene changes in this segment first
        segment_frames = _extract_scene_changes_in_range(
            video_path, output_dir, seg_start, seg_end, max_frames=5
        )

        # If no scene changes, extract evenly spaced frames in this segment
        if len(segment_frames) < 3:
            segment_frames.extend(_extract_evenly_spaced_in_range(
                video_path, output_dir, seg_start, seg_end, count=3
            ))

        all_frames.extend(segment_frames)

    logger.info(f"Extracted {len(all_frames)} frames across {num_segments} video segments")

    # Score all frames for quality
    logger.info(f"Scoring {len(all_frames)} frames for face detection and visual quality...")
    for frame in all_frames:
        scores = _score_frame_quality(frame["file_path"])
        frame.update(scores)
        # Add plan-spec keys: face_score and quality_score normalized to [0, 1]
        frame["face_score"] = round(scores.get("face_score", 0.0) / 100.0, 4)
        frame["quality_score"] = score_frame_quality(frame["file_path"])

    # Sort by composite score (face 0.7 + quality 0.3) descending
    all_frames.sort(
        key=lambda f: f["face_score"] * 0.7 + f["quality_score"] * 0.3,
        reverse=True,
    )

    # Enforce frame diversity using perceptual hashing before taking top N
    effective_max = max_frames if max_frames > 0 else MAX_CANDIDATE_FRAMES
    top_frames = _enforce_frame_diversity(all_frames, effective_max)

    # Re-sort by timestamp for output
    top_frames.sort(key=lambda f: f["timestamp"])

    # Keep original full-res frames for compositing; create 512x288 copies for Claude
    claude_dir = output_dir / "claude_ready"
    claude_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(top_frames):
        src_path = Path(frame["file_path"])
        # Rename original to clean name (keep full resolution)
        fullres_path = output_dir / f"frame_{i:04d}.jpg"
        if src_path.resolve() != fullres_path.resolve():
            src_path.rename(fullres_path)
        # Create small copy for Claude analysis
        claude_path = claude_dir / f"frame_{i:04d}.jpg"
        _resize_frame(fullres_path, claude_path, target_w=512, target_h=288, keep_original=True)
        frame["file_path"] = str(fullres_path)  # Full-res for compositing
        frame["claude_path"] = str(claude_path)  # Small for Claude
        frame["frame_index"] = i

    if not top_frames:
        logger.warning(f"No frames extracted for job {job_id}")
        return []

    first_ts = top_frames[0]["timestamp"]
    last_ts = top_frames[-1]["timestamp"]
    coverage = (last_ts - first_ts) / duration * 100 if duration > 0 else 0

    logger.info(
        f"Extracted {len(top_frames)} candidate frames for job {job_id}. "
        f"Coverage: {first_ts:.1f}s to {last_ts:.1f}s ({coverage:.1f}% of video). "
        f"Avg face_score: {sum(f['face_score'] for f in top_frames) / len(top_frames):.3f}"
    )
    return top_frames


def _resize_frame(src: Path, dest: Path, target_w: int = 512, target_h: int = 288, keep_original: bool = False) -> None:
    """
    Resize a frame to target dimensions using Pillow LANCZOS and save as JPEG.
    """
    try:
        img = Image.open(src)
        img = img.resize((target_w, target_h), Image.LANCZOS)
        img.save(str(dest), "JPEG", quality=90)
        if not keep_original and src.resolve() != dest.resolve():
            try:
                src.unlink()
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"Failed to resize frame {src}: {e}")


def _compute_phash(image_path: str) -> np.ndarray | None:
    """
    Compute perceptual hash (pHash) for an image.
    Returns a 64-bit binary hash as a numpy array, or None on failure.

    Method: resize 32x32 -> grayscale -> DCT -> threshold top-left 8x8.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)
    dct = cv2.dct(resized)
    # Take top-left 8x8 (low-frequency components)
    dct_low = dct[:8, :8]
    # Threshold at median (excluding DC component)
    median_val = np.median(dct_low)
    return (dct_low > median_val).flatten().astype(np.uint8)


def _hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming distance between two binary hash arrays."""
    return int(np.sum(hash1 != hash2))


def _enforce_frame_diversity(frames: list[dict], max_count: int, min_distance: int = 10) -> list[dict]:
    """
    Greedily select diverse frames using perceptual hashing.

    Picks the highest-scored frame first, then only adds frames whose pHash
    Hamming distance is > min_distance from ALL already-selected frames.
    This prevents multiple thumbnails from using the same scene.
    """
    if len(frames) <= max_count:
        return frames

    # Compute hashes for all frames
    for frame in frames:
        frame["_phash"] = _compute_phash(frame["file_path"])

    selected = []
    for frame in frames:  # Already sorted by score descending
        if frame["_phash"] is None:
            # Can't hash — accept if we need more frames
            if len(selected) < max_count:
                selected.append(frame)
            continue

        # Check diversity against all selected frames
        is_diverse = True
        for sel in selected:
            if sel["_phash"] is not None:
                dist = _hamming_distance(frame["_phash"], sel["_phash"])
                if dist <= min_distance:
                    is_diverse = False
                    break

        if is_diverse:
            selected.append(frame)

        if len(selected) >= max_count:
            break

    # If we didn't get enough diverse frames, fill from remaining
    if len(selected) < max_count:
        remaining = [f for f in frames if f not in selected]
        selected.extend(remaining[:max_count - len(selected)])

    # Clean up temporary hash data
    for frame in frames:
        frame.pop("_phash", None)

    return selected


def _extract_scene_changes_in_range(
    video_path: str,
    output_dir: Path,
    start_time: float,
    end_time: float,
    max_frames: int = 3
) -> list[dict]:
    """
    Detect scene changes within a specific time range and extract those frames.
    Returns at most max_frames from this range.
    """
    # Use FFmpeg scdet to find scene change timestamps in range
    detect_cmd = [
        FFMPEG_PATH,
        "-ss", str(start_time),  # Start at this time
        "-t", str(end_time - start_time),  # Duration of segment
        "-i", video_path,
        "-vf", "scdet=threshold=0.3",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        detect_cmd, capture_output=True, text=True, timeout=60
    )

    # Parse scene change timestamps from stderr
    scene_times = []
    for line in result.stderr.split("\n"):
        if "lavfi.scd.time" in line:
            try:
                parts = line.split("lavfi.scd.time:")
                if len(parts) > 1:
                    time_str = parts[1].strip().split()[0]
                    # Time is relative to segment start, add offset
                    t = float(time_str) + start_time
                    if start_time <= t <= end_time:
                        scene_times.append(t)
            except (ValueError, IndexError):
                continue

    # Limit to max_frames per segment
    scene_times = scene_times[:max_frames]
    frames = []

    for t in scene_times:
        # Use timestamp in filename for uniqueness
        output_path = output_dir / f"seg_{int(start_time):04d}_{len(frames):02d}.jpg"
        extract_cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]
        result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and output_path.exists():
            frames.append({
                "timestamp": round(t, 2),
                "file_path": str(output_path),
                "source": "scene_change",
            })

    return frames


def _extract_evenly_spaced_in_range(
    video_path: str,
    output_dir: Path,
    start_time: float,
    end_time: float,
    count: int = 2
) -> list[dict]:
    """
    Extract evenly spaced frames within a specific time range.
    Used as fallback when no scene changes found in a segment.
    """
    duration = end_time - start_time
    interval = duration / (count + 1)
    frames = []

    for i in range(count):
        t = start_time + interval * (i + 1)

        output_path = output_dir / f"even_{int(start_time):04d}_{i:02d}.jpg"
        extract_cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]
        result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and output_path.exists():
            frames.append({
                "timestamp": round(t, 2),
                "file_path": str(output_path),
                "source": "evenly_spaced",
            })

    return frames


def _score_frame_quality(image_path: str) -> dict:
    """
    Score a frame for thumbnail quality based on:
    - Face detection & expression quality (0-100)
    - Visual sharpness (0-100)
    - Lighting/contrast quality (0-100)
    - Composition quality (0-100)

    Returns dict with individual scores and combined score.
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not load image: {image_path}")
        return {
            "face_score": 0,
            "sharpness_score": 0,
            "lighting_score": 0,
            "composition_score": 0,
            "combined_score": 0,
            "has_face": False,
            "expression_type": "neutral",
        }

    # 1. Face detection and expression scoring
    face_score, has_face, face_size, expression_type = _detect_faces_and_expressions(img)

    # 2. Sharpness scoring
    sharpness_score = _calculate_sharpness(img)

    # 3. Lighting/contrast scoring
    lighting_score = _calculate_brightness_contrast(img)

    # 4. Composition scoring
    composition_score = _calculate_composition_score(img, face_size)

    # Combined score with weights (prioritize faces heavily for lifestyle/comedy)
    weights = {
        "face": 0.45,        # Face presence and expression is most important
        "sharpness": 0.25,   # Clear, sharp image
        "lighting": 0.15,    # Good lighting
        "composition": 0.15, # Good framing
    }

    combined_score = (
        face_score * weights["face"] +
        sharpness_score * weights["sharpness"] +
        lighting_score * weights["lighting"] +
        composition_score * weights["composition"]
    )

    return {
        "face_score": round(face_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "lighting_score": round(lighting_score, 2),
        "composition_score": round(composition_score, 2),
        "combined_score": round(combined_score, 2),
        "has_face": has_face,
        "expression_type": expression_type,
    }


def _detect_faces_and_expressions(img: np.ndarray) -> tuple[float, bool, float, str]:
    """
    Detect faces using MediaPipe and score based on:
    - Face presence and size (larger = better for thumbnails)
    - Face clarity and position
    - Expression quality (open mouth, wide eyes = expressive = higher score)

    Returns (score, has_face, relative_face_size, expression_type)
    """
    try:
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detect_faces(img_rgb)

        if not faces:
            return 0.0, False, 0.0, "neutral"

        # Use the largest face (primary subject)
        face = faces[0]
        face_area = face.area_ratio

        # 1. Face size score (larger = better, up to 40% of frame)
        optimal_size = 0.30
        size_score = 100 * (1 - abs(face_area - optimal_size) / optimal_size)
        size_score = max(0, min(100, size_score))

        if face_area > 0.15:
            size_score = min(100, size_score * 1.2)

        # 2. Position score (rule of thirds)
        face_center_x = face.center[0] / w
        face_center_y = face.center[1] / h

        thirds_points = [
            (0.33, 0.33), (0.67, 0.33),
            (0.33, 0.67), (0.67, 0.67),
            (0.5, 0.4),
        ]
        min_dist = min(
            ((face_center_x - px) ** 2 + (face_center_y - py) ** 2) ** 0.5
            for px, py in thirds_points
        )
        position_score = 100 * (1 - min(min_dist, 0.3) / 0.3)

        # 3. Confidence score
        confidence_score = face.confidence * 100

        # 4. Expression bonus (expressive faces score higher)
        # Uses the new combined expressiveness score from FaceResult
        expression_bonus = face.expressiveness * 30

        # 5. Face completeness penalty — reject faces cut off at frame edges
        fx, fy, fw, fh = face.bbox
        completeness_penalty = 0
        # If face bbox starts at/near top of frame, eyes likely cut off
        if fy < fh * 0.15:
            completeness_penalty -= 40
        # If face bbox extends to/near bottom of frame, chin/neck framing
        if (fy + fh) > h * 0.92:
            completeness_penalty -= 15
        # If face is in the very top 20% of frame, bad low-angle shot
        if face.center[1] < h * 0.20:
            completeness_penalty -= 30
        # If face bbox is cut at left/right edges
        if fx < fw * 0.1 or (fx + fw) > w * 0.95:
            completeness_penalty -= 10

        # 6. Motion blur penalty on face region
        face_region = img[fy:fy + fh, fx:fx + fw]
        blur_penalty = _calculate_motion_blur(face_region)

        # 7. Classify expression type based on highest blendshape category
        expression_type = "neutral"
        if face.expression_categories:
            cats = face.expression_categories
            best_cat = max(cats, key=cats.get)
            # Only label as non-neutral if the winning category has a meaningful score
            if cats[best_cat] > 0.02:
                expression_type = best_cat

        # Combined face score
        face_score = (
            size_score * 0.40 +
            position_score * 0.25 +
            confidence_score * 0.15 +
            expression_bonus +
            blur_penalty +
            completeness_penalty
        )

        return min(100, face_score), True, face_area, expression_type

    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return 50.0, False, 0.0, "neutral"


def _calculate_motion_blur(face_img: np.ndarray) -> float:
    """
    Detect motion blur in the face region using Laplacian variance.
    Returns a penalty (negative value) if the face region is blurry,
    or 0 if it is acceptably sharp.

    Args:
        face_img: BGR image cropped to the face bounding box.

    Returns:
        0 if sharp enough, -20 if blurry (Laplacian variance < 50).
    """
    if face_img is None or face_img.size == 0:
        return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return -20
    return 0


def _calculate_sharpness(img: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher variance = sharper image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize to 0-100 scale
    sharpness_score = min(100, (laplacian_var / 500) * 100)

    return sharpness_score


def _calculate_brightness_contrast(img: np.ndarray) -> float:
    """
    Score lighting quality based on brightness distribution and contrast.
    Good thumbnails have good contrast without being over/underexposed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    mean_brightness = np.mean(gray)
    brightness_score = 100 * (1 - abs(mean_brightness - 140) / 140)
    brightness_score = max(0, min(100, brightness_score))

    contrast = np.std(gray)
    contrast_score = 100 * min(contrast / 60, 1.0)

    clipped_dark = np.sum(hist[:10])
    clipped_bright = np.sum(hist[-10:])
    clipping_penalty = (clipped_dark + clipped_bright) * 100

    lighting_score = (
        brightness_score * 0.4 +
        contrast_score * 0.5 -
        clipping_penalty * 0.1
    )

    return max(0, min(100, lighting_score))


def _calculate_composition_score(img: np.ndarray, face_area: float) -> float:
    """
    Score composition quality:
    - Good use of frame space
    - Not too much empty space
    - Color variety and visual interest
    """
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    mean_saturation = np.mean(saturation)
    saturation_score = min(100, (mean_saturation / 128) * 100)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    if 0.05 <= edge_density <= 0.20:
        interest_score = 100
    else:
        interest_score = 100 * (1 - abs(edge_density - 0.125) / 0.125)
        interest_score = max(0, min(100, interest_score))

    subject_bonus = min(30, face_area * 100) if face_area > 0 else 0

    composition_score = (
        saturation_score * 0.4 +
        interest_score * 0.4 +
        subject_bonus * 0.2
    )

    return min(100, composition_score)


def extract_single_frame(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract a single frame at a specific timestamp."""
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract frame at {timestamp}s")
    return output_path
