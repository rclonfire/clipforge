"""
Thumbnail quality gates — squint test and contrast ratio enforcement.

Validates every thumbnail before export:
- Squint test: ensures subject is identifiable at 160px width
- Contrast check: ensures text readability with WCAG 4:1 minimum ratio
- Contrast enforcement: auto-adds outline/shadow when contrast is below 4:1

These gates run after Pillow compositing but before final save.
Failures are logged as warnings and never raise — quality gates are
informational and must not crash thumbnail generation.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum subject coverage fraction at 160px width (comedy thumbnails use tight
# face crops so 15% is generous — a full face fills roughly 20-30% of the frame)
_SQUINT_COVERAGE_THRESHOLD = 0.15

# WCAG 2.1 AA contrast ratio minimum for normal text
_MIN_CONTRAST_RATIO = 4.0


# ---------------------------------------------------------------------------
# Luminance helpers
# ---------------------------------------------------------------------------


def _linearize(c: float) -> float:
    """Convert a single normalised sRGB channel value (0-1) to linear light."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Compute WCAG relative luminance from 0-255 sRGB values."""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    return 0.2126 * _linearize(rn) + 0.7152 * _linearize(gn) + 0.0722 * _linearize(bn)


def compute_contrast_ratio(fg_luminance: float, bg_luminance: float) -> float:
    """
    Compute WCAG contrast ratio between two relative luminance values.

    Formula: (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)

    White on black returns ~21.0.  Identical colours return 1.0.
    """
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Squint test
# ---------------------------------------------------------------------------


def check_squint_test(img: Image.Image, subject_mask: np.ndarray | None) -> dict:
    """
    Validate that the subject is identifiable at 160px width.

    Args:
        img: Full-resolution PIL Image (1280x720).
        subject_mask: Float32 mask (H, W) in [0.0, 1.0] where 1.0 = subject,
                      or None if unavailable.

    Returns:
        dict with keys:
            passed (bool): True if subject coverage >= 15%.
            subject_coverage (float): Fraction of 160px frame covered by subject.
            method (str): "mask" if mask was used, "contour" if fallback.
    """
    try:
        small = img.resize((160, 90), Image.LANCZOS)

        if subject_mask is not None:
            # Resize mask to 160x90 and check coverage
            mask_img = Image.fromarray((subject_mask * 255).astype(np.uint8)).resize(
                (160, 90), Image.LANCZOS
            )
            mask_small = np.array(mask_img) / 255.0
            coverage = float(np.mean(mask_small > 0.5))
            method = "mask"
        else:
            # Fallback: OpenCV contour detection on grayscale thumbnail
            gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
            # Adaptive threshold to isolate foreground
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            total_pixels = gray.shape[0] * gray.shape[1]
            if contours:
                largest_area = max(cv2.contourArea(c) for c in contours)
                coverage = float(largest_area / total_pixels)
            else:
                coverage = 0.0
            method = "contour"

        passed = coverage >= _SQUINT_COVERAGE_THRESHOLD
        if not passed:
            logger.warning(
                f"thumbnail_quality: squint test FAILED — subject coverage "
                f"{coverage:.1%} at 160px (need {_SQUINT_COVERAGE_THRESHOLD:.0%}+). "
                f"Thumbnail may not be identifiable at small sizes. [method={method}]"
            )
        return {"passed": passed, "subject_coverage": coverage, "method": method}

    except Exception as exc:
        logger.debug(f"thumbnail_quality: check_squint_test error — {exc}")
        return {"passed": True, "subject_coverage": 0.0, "method": "error"}


# ---------------------------------------------------------------------------
# Text contrast
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' or '#RGB' hex string to (r, g, b) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def _sample_region_luminance(img: Image.Image, text_position: str) -> float:
    """
    Sample mean relative luminance of the background region corresponding to text_position.

    Crops a representative quadrant of the image and computes the average
    luminance, which approximates what's behind the text.
    """
    w, h = img.size
    half_w, half_h = w // 2, h // 2

    pos = (text_position or "bottom-left").lower()

    if "top" in pos:
        y0, y1 = 0, half_h
    elif "bottom" in pos:
        y0, y1 = half_h, h
    else:
        # center
        y0, y1 = h // 4, 3 * h // 4

    if "left" in pos:
        x0, x1 = 0, half_w
    elif "right" in pos:
        x0, x1 = half_w, w
    else:
        x0, x1 = w // 4, 3 * w // 4

    region = img.crop((x0, y0, x1, y1)).convert("RGB")
    arr = np.array(region, dtype=np.float32) / 255.0

    # Vectorised luminance computation
    r_lin = np.where(arr[:, :, 0] <= 0.04045,
                     arr[:, :, 0] / 12.92,
                     ((arr[:, :, 0] + 0.055) / 1.055) ** 2.4)
    g_lin = np.where(arr[:, :, 1] <= 0.04045,
                     arr[:, :, 1] / 12.92,
                     ((arr[:, :, 1] + 0.055) / 1.055) ** 2.4)
    b_lin = np.where(arr[:, :, 2] <= 0.04045,
                     arr[:, :, 2] / 12.92,
                     ((arr[:, :, 2] + 0.055) / 1.055) ** 2.4)

    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return float(np.mean(luminance))


def check_text_contrast(img: Image.Image, concept: dict) -> dict:
    """
    Measure contrast ratio between text colour and its background region.

    Args:
        img: Composited PIL Image before text has been rendered.
        concept: Concept dict with text_overlay, text_position, text_color keys.

    Returns:
        dict with keys:
            passed (bool): True if ratio >= 4:1 or no text.
            ratio (float | None): Computed WCAG ratio, or None if no text.
            text_color (str): Hex colour string.
            bg_luminance (float): Sampled background luminance.
    """
    try:
        text_overlay = concept.get("text_overlay") or ""
        if not text_overlay.strip():
            return {"passed": True, "ratio": None, "reason": "no text",
                    "text_color": "", "bg_luminance": 0.0}

        text_color_hex = concept.get("text_color") or "#FFFFFF"
        text_position = concept.get("text_position") or "bottom-left"

        r, g, b = _hex_to_rgb(text_color_hex)
        text_luminance = _relative_luminance(r, g, b)
        bg_luminance = _sample_region_luminance(img, text_position)

        ratio = compute_contrast_ratio(text_luminance, bg_luminance)
        passed = ratio >= _MIN_CONTRAST_RATIO

        return {
            "passed": passed,
            "ratio": ratio,
            "text_color": text_color_hex,
            "bg_luminance": bg_luminance,
        }

    except Exception as exc:
        logger.debug(f"thumbnail_quality: check_text_contrast error — {exc}")
        return {"passed": True, "ratio": None, "text_color": "", "bg_luminance": 0.0}


# ---------------------------------------------------------------------------
# Contrast enforcement
# ---------------------------------------------------------------------------


def enforce_text_contrast(img: Image.Image, concept: dict) -> tuple[Image.Image, dict]:
    """
    Auto-fix text contrast if below 4:1 WCAG threshold.

    Called BEFORE the text rendering step. If contrast is already sufficient,
    returns (img, concept) unchanged. If not, modifies concept to add
    outline/shadow treatment so the downstream renderer produces readable text.

    Args:
        img: Composited PIL Image (before text overlay).
        concept: Concept dict; may be mutated if contrast auto-fix is needed.

    Returns:
        (img, concept) tuple — img is always unchanged, concept may be modified.
    """
    try:
        result = check_text_contrast(img, concept)
        ratio = result.get("ratio")

        if ratio is None:
            # No text — nothing to fix
            return img, concept

        if ratio >= _MIN_CONTRAST_RATIO:
            return img, concept

        logger.info(
            f"thumbnail_quality: Auto-fixing text contrast — ratio {ratio:.1f}:1 "
            f"below {_MIN_CONTRAST_RATIO:.0f}:1 threshold, adding outline/shadow"
        )

        # Mutate a shallow copy so callers aren't surprised
        concept = dict(concept)
        concept["text_stroke_width"] = 4
        concept["text_shadow"] = True

        # If text colour is dark on a dark background, force white text
        text_color_hex = concept.get("text_color") or "#FFFFFF"
        r, g, b = _hex_to_rgb(text_color_hex)
        text_luminance = _relative_luminance(r, g, b)
        bg_luminance = result.get("bg_luminance", 0.0)

        if text_luminance < 0.18 and bg_luminance < 0.18:
            concept["text_color"] = "#FFFFFF"
            logger.debug(
                "thumbnail_quality: Dark text on dark background — "
                "overriding text_color to #FFFFFF"
            )

        return img, concept

    except Exception as exc:
        logger.debug(f"thumbnail_quality: enforce_text_contrast error — {exc}")
        return img, concept
