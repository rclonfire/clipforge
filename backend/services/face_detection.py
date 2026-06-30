"""
Shared face detection module using MediaPipe Tasks API (0.10.x).

Replaces Haar Cascade with MediaPipe FaceDetector + FaceLandmarker for
accurate face/expression detection and ImageSegmenter for subject masks.
Used by frame_extraction, thumbnail_generator, and related modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    ImageSegmenter,
    ImageSegmenterOptions,
)

logger = logging.getLogger(__name__)

# Model file paths (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_FACE_DETECTOR_MODEL = _MODELS_DIR / "blaze_face_short_range.tflite"
_FACE_LANDMARKER_MODEL = _MODELS_DIR / "face_landmarker.task"
_SELFIE_SEGMENTER_MODEL = _MODELS_DIR / "selfie_segmenter.tflite"

# Lazy singletons
_face_detector: FaceDetector | None = None
_face_landmarker: FaceLandmarker | None = None
_image_segmenter: ImageSegmenter | None = None


def _get_face_detector() -> FaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(_FACE_DETECTOR_MODEL)),
                min_detection_confidence=0.5,
            )
        )
    return _face_detector


def _get_face_landmarker() -> FaceLandmarker:
    global _face_landmarker
    if _face_landmarker is None:
        _face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_FACE_LANDMARKER_MODEL)),
                num_faces=3,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
            )
        )
    return _face_landmarker


def _get_image_segmenter() -> ImageSegmenter:
    global _image_segmenter
    if _image_segmenter is None:
        _image_segmenter = ImageSegmenter.create_from_options(
            ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(_SELFIE_SEGMENTER_MODEL)),
                output_confidence_masks=True,
            )
        )
    return _image_segmenter


@dataclass
class FaceResult:
    """Detection result for a single face."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center: tuple[int, int]          # (cx, cy)
    area_ratio: float                # face area / image area
    confidence: float                # detection confidence [0, 1]
    mouth_openness: float = 0.0      # 0 = closed, 1 = wide open
    eye_openness: float = 0.0        # 0 = closed, 1 = wide open
    expressiveness: float = 0.0      # 0 = neutral, 1 = highly expressive
    expression_categories: dict[str, float] | None = None  # per-category scores for classification


def detect_faces(img_rgb: np.ndarray) -> list[FaceResult]:
    """
    Detect faces using MediaPipe FaceDetector + FaceLandmarker for expression scoring.

    Args:
        img_rgb: Image in RGB format (H, W, 3), uint8.

    Returns:
        List of FaceResult sorted by area_ratio descending (largest first).
    """
    h, w = img_rgb.shape[:2]
    image_area = h * w

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # --- Phase 1: Face Detection (bounding boxes + confidence) ---
    detector = _get_face_detector()
    det_result = detector.detect(mp_image)

    if not det_result.detections:
        return []

    faces: list[FaceResult] = []

    for detection in det_result.detections:
        bb = detection.bounding_box
        x = max(0, bb.origin_x)
        y = max(0, bb.origin_y)
        fw = min(bb.width, w - x)
        fh = min(bb.height, h - y)

        if fw <= 0 or fh <= 0:
            continue

        area_ratio = (fw * fh) / image_area
        center = (x + fw // 2, y + fh // 2)
        confidence = detection.categories[0].score if detection.categories else 0.0

        faces.append(FaceResult(
            bbox=(x, y, fw, fh),
            center=center,
            area_ratio=area_ratio,
            confidence=confidence,
        ))

    # --- Phase 2: FaceLandmarker (expression scoring via blendshapes) ---
    try:
        landmarker = _get_face_landmarker()
        lm_result = landmarker.detect(mp_image)

        if lm_result.face_blendshapes:
            for i, blendshapes in enumerate(lm_result.face_blendshapes):
                if i >= len(faces):
                    break

                # Build a lookup of blendshape name -> score
                bs = {b.category_name: b.score for b in blendshapes}

                # Mouth openness from jawOpen blendshape
                mouth_openness = min(1.0, bs.get("jawOpen", 0.0) * 2.0)

                # Eye openness: inverse of eyeBlink (1 - blink = open)
                left_blink = bs.get("eyeBlinkLeft", 0.0)
                right_blink = bs.get("eyeBlinkRight", 0.0)
                avg_blink = (left_blink + right_blink) / 2.0
                eye_openness = max(0.0, 1.0 - avg_blink)

                # Bonus: wide eyes (eyeWide blendshape)
                left_wide = bs.get("eyeWideLeft", 0.0)
                right_wide = bs.get("eyeWideRight", 0.0)
                wide_bonus = (left_wide + right_wide) / 2.0
                eye_openness = min(1.0, eye_openness + wide_bonus * 0.5)

                # Expressiveness: weighted combination of high-value blendshapes
                # Surprise-related (weight 0.3)
                surprise_score = (
                    bs.get("jawOpen", 0.0) +
                    bs.get("eyeWideLeft", 0.0) +
                    bs.get("eyeWideRight", 0.0)
                ) * 0.3

                # Smile/happiness (weight 0.25)
                smile_score = (
                    bs.get("mouthSmileLeft", 0.0) +
                    bs.get("mouthSmileRight", 0.0)
                ) * 0.25

                # Brow movement (weight 0.2)
                brow_score = (
                    bs.get("browInnerUp", 0.0) +
                    bs.get("browDownLeft", 0.0) +
                    bs.get("browDownRight", 0.0)
                ) * 0.2

                # Mouth shapes (weight 0.15)
                mouth_shape_score = (
                    bs.get("mouthFunnel", 0.0) +
                    bs.get("mouthPucker", 0.0)
                ) * 0.15

                # Other/comedic (weight 0.1)
                other_score = (
                    bs.get("cheekPuff", 0.0)
                ) * 0.1

                expressiveness = min(
                    1.0,
                    surprise_score + smile_score + brow_score +
                    mouth_shape_score + other_score,
                )

                faces[i].mouth_openness = round(mouth_openness, 3)
                faces[i].eye_openness = round(eye_openness, 3)
                faces[i].expressiveness = round(expressiveness, 3)
                faces[i].expression_categories = {
                    "surprise": round(surprise_score, 3),
                    "happy": round(smile_score, 3),
                    "intense": round(brow_score, 3),
                    "talking": round(mouth_shape_score, 3),
                }
    except Exception as e:
        logger.debug(f"FaceLandmarker expression scoring failed (non-critical): {e}")

    # Sort by area descending (largest face first = primary subject)
    faces.sort(key=lambda f: f.area_ratio, reverse=True)
    return faces


def score_frame_quality(image_path: str) -> float:
    """
    Score a frame's sharpness quality using Laplacian variance.

    Returns Laplacian variance / 1000, clamped to [0.0, 1.0].
    A score of 0.0 means the image could not be loaded or is completely blurry.
    A score of 1.0 means the image has a Laplacian variance >= 1000 (very sharp).

    Args:
        image_path: Absolute path to the JPEG/PNG frame file.

    Returns:
        Float quality score in [0.0, 1.0].
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = lap_var / 1000.0
    return float(min(1.0, max(0.0, score)))


def create_subject_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Create a pixel-accurate person segmentation mask using MediaPipe ImageSegmenter.

    Args:
        img_rgb: Image in RGB format (H, W, 3), uint8.

    Returns:
        Float mask (H, W) in range [0.0, 1.0] where 1.0 = person, 0.0 = background.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    segmenter = _get_image_segmenter()
    result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    # The selfie segmenter returns a single confidence mask
    # confidence_masks[0] is the person mask
    mask = result.confidence_masks[0].numpy_view().copy()
    mask = mask.astype(np.float32)

    # Smooth edges with a larger kernel to avoid hard outlines
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    # Soft threshold: push towards 0/1 but keep a gentle gradient at edges
    mask = np.where(mask > 0.6, 1.0, np.where(mask < 0.2, 0.0, mask)).astype(np.float32)
    # Final smooth to ensure no hard edge artifacts
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return np.clip(mask, 0.0, 1.0)
