"""
Local frame enhancement pipeline. Runs after frame extraction, before Claude selection.
All processing is local (OpenCV) -- no API cost.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from backend.services.face_detection import create_subject_mask, detect_faces

logger = logging.getLogger(__name__)


def isolate_subject(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Use rembg (birefnet-general model) to isolate the subject from the background.

    The rembg import is deferred inside the function to avoid triggering model
    downloads or onnxruntime initialisation at module import time.

    Args:
        img_bgr: Input image in BGR format (H, W, 3), uint8.

    Returns:
        Float mask (H, W) in [0.0, 1.0] where 1.0 = foreground subject.
        Returns None on any failure so the caller can degrade gracefully.
    """
    try:
        from rembg import remove  # lazy import -- avoids model load at startup

        # rembg expects PIL Image in RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Remove background -- returns RGBA PIL Image with transparent background
        result = remove(pil_img, model_name="birefnet-general")

        # Extract alpha channel as float32 mask in [0.0, 1.0]
        alpha = np.array(result)[:, :, 3].astype(np.float32) / 255.0

        # Smooth mask edges with 2px gaussian blur (kernel 5x5, sigma=2.0)
        # Per user decision: soft edges prevent harsh cutout artifacts
        alpha = cv2.GaussianBlur(alpha, (5, 5), 2.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha

    except Exception as e:
        logger.warning(f"isolate_subject failed (graceful fallback): {e}")
        return None


def enhance_frame(frame_path: str, output_path: str | None = None) -> str:
    """
    Apply local enhancement pipeline to a single frame:
      1. CLAHE on luminance channel (contrast boost without color distortion)
      2. Unsharp mask sharpening (alpha=1.5)
      3. Saturation boost (1.3x)
      4. Background desaturation using subject mask (35% saturation in background)

    If enhancement fails, the original file is left untouched and the raw path
    is returned (caller should treat this as graceful fallback).

    Args:
        frame_path: Absolute path to the source JPEG frame.
        output_path: Where to write the enhanced image. If None, overwrites frame_path.

    Returns:
        The path where the enhanced (or original) image lives.
    """
    dest = output_path if output_path is not None else frame_path

    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"cv2.imread could not load: {frame_path}")

    # -----------------------------------------------------------------------
    # Step 1: CLAHE on luminance channel only (prevents color distortion)
    # Convert BGR -> LAB, apply CLAHE to L, merge back, convert LAB -> BGR
    # -----------------------------------------------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -----------------------------------------------------------------------
    # Step 2: Unsharp mask sharpening
    # blurred = GaussianBlur(img, sigma=3); result = 1.5*img - 0.5*blurred
    # -----------------------------------------------------------------------
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # -----------------------------------------------------------------------
    # Step 3: Saturation boost (1.3x)
    # Convert BGR -> HSV (float32), multiply S by 1.3, clip, convert back
    # -----------------------------------------------------------------------
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # -----------------------------------------------------------------------
    # Step 4: Background desaturation using subject mask
    # Subject regions keep full saturation; background desaturated to ~35%
    # -----------------------------------------------------------------------
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    mask = create_subject_mask(img_rgb)  # (H, W) float32 [0.0, 1.0]

    # Smooth mask edges with 2px gaussian blur before blending
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)
    mask = np.clip(mask, 0.0, 1.0)

    # Work in HSV for selective saturation control
    hsv_final = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_channel = hsv_final[:, :, 1]

    # Desaturated version of S: 35% of current (= desaturated ~65% relative to original)
    s_desat = s_channel * 0.35

    # Soft blend: subject areas keep full S, background gets desaturated S
    # mask=1.0 -> subject (keep full), mask=0.0 -> background (desaturate)
    hsv_final[:, :, 1] = mask * s_channel + (1.0 - mask) * s_desat
    hsv_final[:, :, 1] = np.clip(hsv_final[:, :, 1], 0, 255)
    enhanced = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # -----------------------------------------------------------------------
    # Step 5: Subject isolation via rembg + composite onto darkened background
    # Only runs when a face is detected (area_ratio > 0.03).
    # Failures fall back to the already-enhanced frame (no exception propagated).
    # -----------------------------------------------------------------------
    try:
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        faces = detect_faces(img_rgb)

        if faces and faces[0].area_ratio > 0.03:
            rembg_mask = isolate_subject(enhanced)

            if rembg_mask is not None and np.mean(rembg_mask > 0.5) > 0.05:
                # Darken the already-desaturated background by 30%
                bg = (enhanced.astype(np.float32) * 0.7).astype(np.uint8)

                # Composite: subject pixels from enhanced, background pixels from bg
                mask_3ch = np.stack([rembg_mask] * 3, axis=-1)
                enhanced = (
                    enhanced.astype(np.float32) * mask_3ch
                    + bg.astype(np.float32) * (1.0 - mask_3ch)
                ).astype(np.uint8)
            else:
                logger.debug(
                    "rembg mask coverage too low or None, skipping isolation"
                )
        else:
            logger.debug("No face detected (or face too small), skipping subject isolation")
    except Exception as e:
        logger.warning(f"Subject isolation step failed, using enhanced-only frame: {e}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    cv2.imwrite(dest, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dest


def enhance_frames_batch(frames: list[dict]) -> list[dict]:
    """
    Enhance all candidate frames in-place (both full-res and Claude-ready copies).

    Iterates over the frames list produced by extract_candidate_frames(). For each
    frame, enhances frame["file_path"] (full-res) and frame["claude_path"] (512x288).
    Enhancement is applied in-place so downstream stages see better pixels at the
    same file paths -- no path changes needed.

    Failed frames are skipped with a warning; raw frame passes through unchanged.

    Args:
        frames: List of frame dicts from extract_candidate_frames().
                Each dict must have "file_path" and "claude_path" keys.

    Returns:
        The same frames list (file paths unchanged, content enhanced).
    """
    success_count = 0
    fail_count = 0

    for frame in frames:
        for path_key in ("file_path", "claude_path"):
            frame_path = frame.get(path_key)
            if not frame_path:
                continue
            try:
                enhance_frame(frame_path)
                success_count += 1
            except Exception as e:
                fail_count += 1
                logger.warning(f"Enhancement failed for {frame_path}: {e}")

    logger.info(
        f"Enhanced {success_count}/{len(frames) * 2} frame copies "
        f"({fail_count} fallback to raw)"
    )
    return frames
