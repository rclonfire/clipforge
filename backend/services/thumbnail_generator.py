from __future__ import annotations

import base64
import json
import logging
import math
import random
import string
from pathlib import Path

import anthropic
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops

from backend.config import (
    ANTHROPIC_API_KEY,
    THUMBNAILS_DIR,
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
    FONTS_DIR,
)
from backend.prompts.thumbnail_system import THUMBNAIL_SYSTEM_PROMPT
from backend.services.face_detection import detect_faces, create_subject_mask
from backend.services.thumbnail_quality import (
    check_squint_test,
    enforce_text_contrast,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font system — each font_style targets a different visual feel.
# Claude picks the style; the renderer loads the first available font.
# "width_factor" scales base_size to keep text from overflowing: narrow
# fonts can go bigger, wide fonts need to go smaller.
# ---------------------------------------------------------------------------
# Font styles — tuned for natural hair, faith-based, and lifestyle/beauty
# content based on what top-performing creators in the space actually use.
# ---------------------------------------------------------------------------
FONT_STYLES = {
    "condensed": {
        # Tall, narrow — allows massive font sizes. Best for 1-3 word hooks.
        # Used universally across all YouTube niches for maximum impact.
        "paths": [
            FONTS_DIR / "BebasNeue-Regular.ttf",
            FONTS_DIR / "Anton-Regular.ttf",
            FONTS_DIR / "Oswald-Bold.ttf",
        ],
        "width_factor": 1.15,   # can be bigger (narrow glyphs)
        "tracking": 4,          # extra letter spacing in px
    },
    "heavy": {
        # Wide, ultra-bold geometric — polished, modern lifestyle feel.
        # Montserrat + Poppins: the go-to fonts for beauty & lifestyle creators.
        "paths": [
            FONTS_DIR / "Montserrat-Black.ttf",
            FONTS_DIR / "Poppins-Black.ttf",
            FONTS_DIR / "Montserrat-ExtraBold.ttf",
        ],
        "width_factor": 0.72,   # must be smaller (wide glyphs)
        "tracking": 2,
    },
    "elegant": {
        # Bold display serif — sophisticated, editorial, beauty/faith feel.
        # Abril Fatface: the fashion-editorial font that dominates beauty thumbnails.
        # Playfair Display: refined serif used heavily in faith + lifestyle content.
        "paths": [
            FONTS_DIR / "AbrilFatface-Regular.ttf",
            FONTS_DIR / "PlayfairDisplay.ttf",
        ],
        "width_factor": 0.78,   # serifs take horizontal space
        "tracking": 2,
    },
    "playful": {
        # Rounded, warm, fun — challenge / reaction / lifestyle content.
        # Luckiest Guy: warmer and more inviting than Bangers (better for beauty niche).
        # Bangers: comic-book energy for high-energy moments.
        "paths": [
            FONTS_DIR / "LuckiestGuy-Regular.ttf",
            FONTS_DIR / "Bangers-Regular.ttf",
            FONTS_DIR / "Anton-Regular.ttf",
        ],
        "width_factor": 0.82,
        "tracking": 3,
    },
    "clean": {
        # Modern, friendly, highly legible — the approachable all-rounder.
        # Poppins: geometric sans-serif loved by lifestyle/beauty/health creators.
        "paths": [
            FONTS_DIR / "Poppins-Bold.ttf",
            FONTS_DIR / "Poppins-Black.ttf",
            FONTS_DIR / "Montserrat-Bold.ttf",
        ],
        "width_factor": 0.75,
        "tracking": 1,
    },
    "classic": {
        # Impact / system bold — maximum readability fallback.
        "paths": [
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            FONTS_DIR / "Oswald-Bold.ttf",
            FONTS_DIR / "Anton-Regular.ttf",
        ],
        "width_factor": 0.95,
        "tracking": 2,
    },
}

# Backwards-compat aliases
FONT_STYLES["ultra"] = FONT_STYLES["condensed"]
FONT_STYLES["bold"] = FONT_STYLES["heavy"]
FONT_STYLES["impact"] = FONT_STYLES["classic"]
FONT_STYLES["serif"] = FONT_STYLES["elegant"]
FONT_STYLES["modern"] = FONT_STYLES["clean"]

# Default style
_DEFAULT_FONT_STYLE = "condensed"

# ---------------------------------------------------------------------------
# Emoji vocabulary — key names → unicode characters.
# Rendered via Apple Color Emoji font on macOS.
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "fire": "\U0001F525",       # 🔥
    "eyes": "\U0001F440",       # 👀
    "heart": "\u2764\uFE0F",    # ❤️
    "hearts": "\U0001F495",     # 💕
    "question": "\u2753",       # ❓
    "skull": "\U0001F480",      # 💀
    "scream": "\U0001F631",     # 😱
    "lightning": "\u26A1",      # ⚡
    "sparkles": "\u2728",       # ✨
    "hundred": "\U0001F4AF",    # 💯
}

EMOJI_FONT_PATHS = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
]


def _get_font(style: str = "condensed", size: int = 64) -> ImageFont.FreeTypeFont:
    """Load a font from the specified style's fallback chain."""
    info = FONT_STYLES.get(style, FONT_STYLES[_DEFAULT_FONT_STYLE])
    for path in info["paths"]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_font_style_info(style: str) -> dict:
    """Get width_factor and tracking for a font style."""
    return FONT_STYLES.get(style, FONT_STYLES[_DEFAULT_FONT_STYLE])


def _frame_to_base64(frame_path: str) -> str:
    """Convert a frame image to base64 for Claude API."""
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def analyze_frames_with_claude(
    transcript: str, frames: list, video_title: str = ""
) -> list:
    """
    Send frames and transcript to Claude for thumbnail concept generation.
    Returns list of thumbnail concepts.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Sort frames by combined_score descending and take top 12
    sorted_frames = sorted(
        frames,
        key=lambda f: f.get("combined_score", 0),
        reverse=True,
    )
    frames_to_send = sorted_frames[:12]

    # Build mapping: presentation index -> original frame_index
    # so we can translate Claude's returned indices back
    presentation_to_original: dict[int, int] = {}
    for i, frame in enumerate(frames_to_send):
        presentation_to_original[i] = frame["frame_index"]

    content = [
        {
            "type": "text",
            "text": (
                f"Video title: {video_title}\n\n"
                f"Video transcript:\n{transcript[:8000]}\n\n"
                f"Here are the top {len(frames_to_send)} candidate frames from the video, "
                f"ranked by quality score. Each image is labeled with its frame_index "
                f"and quality metrics.\n\n"
                f"Analyze each frame for face expressions, composition, and thumbnail potential. "
                f"Pick the BEST frames — ones with clear faces, strong emotions, good composition."
            ),
        },
    ]

    for i, frame in enumerate(frames_to_send):
        # Add metadata text block before each frame image
        content.append({
            "type": "text",
            "text": (
                f"Frame {i} (index={frame['frame_index']}): "
                f"timestamp={frame['timestamp']:.1f}s, "
                f"face_score={frame.get('face_score', 0):.0f}, "
                f"sharpness={frame.get('sharpness_score', 0):.0f}, "
                f"expression={frame.get('expression_type', 'unknown')}"
            ),
        })
        # Use claude_path (512x288) for API to save tokens; fall back to file_path
        claude_frame = frame.get("claude_path", frame["file_path"])
        b64 = _frame_to_base64(claude_frame)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })

    content.append({
        "type": "text",
        "text": (
            "Design 3-5 high-CTR thumbnail concepts. For each concept:\n"
            "- Pick the best frame_index (use the index= value shown above; "
            "EACH concept MUST use a DIFFERENT frame_index)\n"
            "- Write SHORT, PUNCHY text_overlay (2-5 words MAX, ALL CAPS) — must be SPECIFIC to this video\n"
            "- Choose text_position from: top-left, top-right, bottom-left, bottom-right, center-left, center-right\n"
            "- Set text_color (use bright yellows #FFD700, whites #FFFFFF, or reds #FF3333)\n"
            "- Set text_stroke_color (usually #000000)\n"
            "- Set background_treatment: 'blur' (default), 'darken', or 'none'\n"
            "- Set text_hierarchy: 'large' (default), 'medium', or 'small'\n"
            "- Set font_style: 'condensed' (tall narrow), 'heavy' (wide bold), 'elegant' (serif/editorial), 'playful' (rounded), 'clean' (modern friendly), or 'classic'\n"
            "- OPTIONAL: Set highlight_word (ONE word to accent) and highlight_color (hex)\n"
            "- OPTIONAL: Set arrow_overlay {enabled, color, style, target} — YouTube-style arrow pointing at face/subject (max 1 per set)\n"
            "- OPTIONAL: Set emoji_overlays [{emoji, position}] — decorative emojis like fire, eyes, heart, skull (max 2 concepts with emojis)\n"
            "- OPTIONAL: Set quote_style: true when text reads like spoken dialogue\n"
            "- Add style_notes (e.g. 'authentic', 'vibrant', 'cinematic', 'clean', 'warm')\n"
            "- Give reasoning for why this will get clicks\n"
            "- Rate estimated_ctr_tier as 'high', 'medium', or 'low'\n\n"
            "Return JSON with key 'concepts'.\n\n"
            "CRITICAL RULES:\n"
            "1. Each concept MUST use a DIFFERENT frame_index.\n"
            "2. Text must be SPECIFIC to this video — no generic filler like 'WAIT...' or 'NO WAY'.\n"
            "3. Vary font_style across concepts for visual variety.\n"
            "4. At least ONE concept with no text (text_overlay: null).\n"
            "5. Use arrow_overlay on at most 1 concept. Use emoji_overlays on at most 2 concepts."
        ),
    })

    logger.info(f"Sending {len(frames_to_send)} frames to Claude for thumbnail analysis")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=THUMBNAIL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    response_text = response.content[0].text.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
        response_text = "\n".join(lines[1:end_idx])

    concepts = json.loads(response_text)

    if isinstance(concepts, dict) and "concepts" in concepts:
        concept_list = concepts["concepts"]
    elif isinstance(concepts, list):
        concept_list = concepts
    else:
        concept_list = [concepts]

    # Translate Claude's returned frame_index values back to original indices.
    # Claude may return the presentation-order index (0..11) instead of the
    # original frame_index we showed in the metadata.  Build a reverse lookup
    # from original indices so we can detect which case we're in.
    original_indices = set(presentation_to_original.values())
    for concept in concept_list:
        fi = concept.get("frame_index")
        if fi is None:
            continue
        if fi in presentation_to_original and fi not in original_indices:
            # Claude used the presentation index — map it back
            concept["frame_index"] = presentation_to_original[fi]
        # If fi is already in original_indices, Claude used the correct
        # original index (from the metadata label), so no mapping needed.

    return concept_list


def _generate_local_concepts(frames: list, video_title: str = "") -> list:
    """
    Generate thumbnail concepts locally without any API call.
    Picks top 3 frames by combined_score and creates simple styling concepts.
    """
    if not frames:
        return []

    sorted_frames = sorted(
        frames,
        key=lambda f: f.get("combined_score", f.get("face_score", 0) + f.get("quality_score", 0)),
        reverse=True,
    )

    # Take top 3 unique frames
    top_frames = sorted_frames[:3]

    styles = [
        {
            "text_color": "#FFD700",
            "text_stroke_color": "#000000",
            "background_treatment": "darken",
            "text_hierarchy": "large",
            "font_style": "heavy",
            "text_position": "bottom-left",
            "style_notes": "bold, vibrant, high energy",
        },
        {
            "text_color": "#FFFFFF",
            "text_stroke_color": "#000000",
            "background_treatment": "blur",
            "text_hierarchy": "large",
            "font_style": "condensed",
            "text_position": "top-right",
            "style_notes": "clean, cinematic",
        },
        {
            "text_overlay": None,
            "background_treatment": "none",
            "style_notes": "no text, frame speaks for itself",
        },
    ]

    # Generate short text from video title
    title_words = video_title.upper().split() if video_title else []
    short_text = " ".join(title_words[:4]) if title_words else None

    concepts = []
    for i, frame in enumerate(top_frames):
        concept = {
            "frame_index": frame["frame_index"],
            "text_overlay": short_text if i < 2 else None,
            "reasoning": f"Frame {frame['frame_index']} — face_score={frame.get('face_score', 0):.2f}, quality={frame.get('quality_score', 0):.2f}",
            "estimated_ctr_tier": "high" if i == 0 else "medium",
        }
        concept.update(styles[i] if i < len(styles) else styles[-1])
        concepts.append(concept)

    return concepts


# ---------------------------------------------------------------------------
# Arrow overlay (YouTube-style attention arrow)
# ---------------------------------------------------------------------------

def _arrowhead_polygon(
    tip: tuple[int, int],
    angle: float,
    head_length: int = 80,
    head_width: int = 60,
) -> list[tuple[int, int]]:
    """Compute 3 vertices of an arrowhead triangle pointing at *tip*."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    px, py = -dy, dx  # perpendicular

    base_x = tip[0] - dx * head_length
    base_y = tip[1] - dy * head_length
    left = (int(base_x + px * head_width / 2), int(base_y + py * head_width / 2))
    right = (int(base_x - px * head_width / 2), int(base_y - py * head_width / 2))
    return [tip, left, right]


def _add_arrow_overlay(
    img: Image.Image,
    color: str = "#FF3333",
    style: str = "straight",
    target: str = "face",
    subject_mask: np.ndarray | None = None,
) -> Image.Image:
    """Draw a bold YouTube-style arrow pointing at the subject."""
    w, h = img.size

    # --- find arrow target point ---
    target_pt = None
    try:
        faces = detect_faces(np.array(img))
        if faces and target == "face":
            target_pt = faces[0].center
    except Exception:
        pass

    if target_pt is None and subject_mask is not None:
        ys, xs = np.where(subject_mask > 0.5)
        if len(xs) > 0:
            target_pt = (int(np.mean(xs)), int(np.mean(ys)))

    if target_pt is None:
        target_pt = (w // 2, h // 2)

    # --- choose arrow origin (farthest corner, inset 15%) ---
    corners = [
        (int(w * 0.12), int(h * 0.12)),
        (int(w * 0.88), int(h * 0.12)),
        (int(w * 0.12), int(h * 0.88)),
        (int(w * 0.88), int(h * 0.88)),
    ]
    origin = max(corners, key=lambda c: math.hypot(c[0] - target_pt[0], c[1] - target_pt[1]))

    # Arrow tip stops ~50px from target
    dx = target_pt[0] - origin[0]
    dy = target_pt[1] - origin[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return img
    tip_offset = 50
    tip = (int(target_pt[0] - dx / dist * tip_offset),
           int(target_pt[1] - dy / dist * tip_offset))

    angle = math.atan2(dy, dx)

    # Scale arrow sizes to image
    shaft_width = max(12, int(w * 0.014))
    head_length = max(50, int(w * 0.06))
    head_width = max(40, int(w * 0.048))

    # Shaft end: pull back from tip by head_length so shaft meets arrowhead base
    shaft_end = (int(tip[0] - math.cos(angle) * head_length * 0.5),
                 int(tip[1] - math.sin(angle) * head_length * 0.5))

    # Parse color
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    arrow_fill = (r, g, b, 255)
    outline_fill = (0, 0, 0, 255)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    head_pts = _arrowhead_polygon(tip, angle, head_length, head_width)
    outline_head = _arrowhead_polygon(tip, angle, head_length + 10, head_width + 12)

    if style == "curved":
        # Bezier curve: offset control point perpendicular to midpoint
        mid = ((origin[0] + tip[0]) / 2, (origin[1] + tip[1]) / 2)
        perp_offset = dist * 0.25
        ctrl = (int(mid[0] - math.sin(angle) * perp_offset),
                int(mid[1] + math.cos(angle) * perp_offset))

        # Sample bezier
        pts = []
        for t in [i / 40 for i in range(41)]:
            bx = (1 - t)**2 * origin[0] + 2 * (1 - t) * t * ctrl[0] + t**2 * shaft_end[0]
            by = (1 - t)**2 * origin[1] + 2 * (1 - t) * t * ctrl[1] + t**2 * shaft_end[1]
            pts.append((int(bx), int(by)))

        # Outline
        draw.line(pts, fill=outline_fill, width=shaft_width + 10, joint="curve")
        # Colored shaft
        draw.line(pts, fill=arrow_fill, width=shaft_width, joint="curve")
    else:
        # Straight shaft
        draw.line([origin, shaft_end], fill=outline_fill, width=shaft_width + 10)
        draw.line([origin, shaft_end], fill=arrow_fill, width=shaft_width)

    # Arrowhead (outline then fill)
    draw.polygon(outline_head, fill=outline_fill)
    draw.polygon(head_pts, fill=arrow_fill)

    # Glow / drop shadow
    glow = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    glow_arr = np.array(glow).astype(np.float32)
    glow_arr[:, :, 3] = np.clip(glow_arr[:, :, 3] * 0.35, 0, 255)
    glow = Image.fromarray(glow_arr.astype(np.uint8))

    canvas = img.convert("RGBA")
    canvas = Image.alpha_composite(canvas, glow)
    canvas = Image.alpha_composite(canvas, overlay)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Emoji overlay
# ---------------------------------------------------------------------------

def _render_emoji_image(emoji_char: str, size: int = 120) -> Image.Image | None:
    """Render a single emoji to an RGBA image using the system emoji font."""
    for path in EMOJI_FONT_PATHS:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                canvas = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((0, 0), emoji_char, font=font, embedded_color=True)
                bbox = canvas.getbbox()
                if bbox:
                    return canvas.crop(bbox)
            except Exception as e:
                logger.debug(f"Emoji font rendering failed: {e}")
                continue
    return None


def _compute_emoji_position(
    position: str,
    img_size: tuple[int, int],
    emoji_size: tuple[int, int],
    face_bboxes: list | None = None,
    index: int = 0,
) -> tuple[int, int]:
    """Calculate where to place an emoji on the thumbnail."""
    w, h = img_size
    ew, eh = emoji_size
    margin = 40

    if position == "near-face" and face_bboxes:
        fx, fy, fw, fh = face_bboxes[0]
        offsets = [
            (fw + 15, -fh // 3),        # right of face
            (-ew - 15, -fh // 3),        # left of face
            (fw // 3, -fh - 10),         # above face
            (fw // 4, fh + 10),          # below face
        ]
        ox, oy = offsets[index % len(offsets)]
        x = max(margin, min(w - ew - margin, fx + ox))
        y = max(margin, min(h - eh - margin, fy + oy))
        return (x, y)
    elif position == "top-left":
        return (margin, margin)
    elif position == "top-right":
        return (w - ew - margin, margin)
    elif position == "bottom-left":
        return (margin, h - eh - margin)
    elif position == "bottom-right":
        return (w - ew - margin, h - eh - margin)
    else:
        # Default: top-right area
        return (w - ew - margin, margin)


def _add_emoji_overlays(
    img: Image.Image,
    emoji_specs: list[dict],
    face_bboxes: list | None = None,
) -> Image.Image:
    """Add emoji overlays to the thumbnail."""
    if not emoji_specs:
        return img

    canvas = img.convert("RGBA")
    w, h = img.size
    emoji_size = max(60, int(w * 0.09))  # ~115px at 1280w

    for i, spec in enumerate(emoji_specs[:3]):
        emoji_key = spec.get("emoji", "")
        char = EMOJI_MAP.get(emoji_key)
        if not char:
            continue

        emoji_img = _render_emoji_image(char, emoji_size)
        if emoji_img is None:
            continue

        # Slight random rotation for natural look
        rotation = random.randint(-15, 15)
        emoji_img = emoji_img.rotate(rotation, expand=True, resample=Image.BICUBIC)

        ew, eh = emoji_img.size
        pos = _compute_emoji_position(
            spec.get("position", "near-face"),
            (w, h), (ew, eh),
            face_bboxes=face_bboxes,
            index=i,
        )

        # Drop shadow
        shadow = Image.new("RGBA", (ew + 20, eh + 20), (0, 0, 0, 0))
        shadow.paste(emoji_img, (10, 10), emoji_img)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
        shadow_arr = np.array(shadow).astype(np.float32)
        shadow_arr[:, :, 3] = np.clip(shadow_arr[:, :, 3] * 0.4, 0, 255)
        shadow = Image.fromarray(shadow_arr.astype(np.uint8))

        # Composite shadow then emoji
        sx, sy = max(0, pos[0] - 10), max(0, pos[1] - 10)
        # Ensure we don't go out of bounds
        if sx + shadow.width > w:
            sx = w - shadow.width
        if sy + shadow.height > h:
            sy = h - shadow.height
        if sx >= 0 and sy >= 0:
            canvas.paste(shadow, (sx, sy), shadow)
        canvas.paste(emoji_img, pos, emoji_img)

    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Thumbnail generation pipeline
# ---------------------------------------------------------------------------

def generate_thumbnail(
    frame_path: str,
    concept: dict,
    job_id: str,
    thumbnail_id: str,
) -> str:
    """
    Generate a professional, high-CTR thumbnail.
    Pipeline: crop -> cache mask -> background blur -> color grade -> subject boost -> text -> border -> sharpen.
    """
    output_dir = THUMBNAILS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{thumbnail_id}.jpg"

    # Open and crop with face-aware framing
    img = Image.open(frame_path).convert("RGB")
    img = _smart_crop(img, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    style_notes = (concept.get("style_notes") or "").lower()

    # Cache subject mask once (avoids 2-3 redundant MediaPipe inferences)
    img_rgb = np.array(img)
    subject_mask = create_subject_mask(img_rgb)

    # === Step 1: Background blur / subject isolation ===
    # Safety: if subject mask covers very little of the frame (wide shot, no close-up),
    # heavy blur would destroy the image. Scale down or skip blur based on coverage.
    mask_coverage = float(np.mean(subject_mask > 0.5)) if subject_mask is not None else 0.0
    bg_treatment = (concept.get("background_treatment") or "blur").lower()
    if mask_coverage < 0.08:
        # Tiny subject (< 8% of frame) — blur would ruin the image, skip entirely
        bg_treatment = "none"
        logger.debug(f"Skipping background blur: mask coverage {mask_coverage:.1%} too low")
    elif mask_coverage < 0.20:
        # Small subject — use gentle blur instead of heavy blur
        if bg_treatment == "blur":
            img = _apply_background_blur(img, blur_radius=18, darken=0.80, mask=subject_mask)
            bg_treatment = "done"  # mark as already applied

    if bg_treatment == "blur":
        img = _apply_background_blur(img, blur_radius=45, darken=0.60, mask=subject_mask)
    elif bg_treatment == "darken":
        img = _apply_background_blur(img, blur_radius=0, darken=0.45, mask=subject_mask)

    # === Step 2: Color grading ===
    if "mrbeast" in style_notes or "ultra bright" in style_notes:
        img = _apply_color_grade(img, "mrbeast")
    elif "dark mode" in style_notes or "neon" in style_notes:
        img = _apply_color_grade(img, "dark")
    elif "warm" in style_notes:
        img = _apply_color_grade(img, "warm")
    elif "clean" in style_notes or "minimal" in style_notes:
        img = _apply_color_grade(img, "clean")
    elif "vibrant" in style_notes:
        img = _apply_color_grade(img, "vibrant")
    elif "cinematic" in style_notes:
        img = _apply_color_grade(img, "cinematic")
    else:
        img = _apply_color_grade(img, "authentic")

    # === Step 3: Subject saturation boost (stacks with bg blur) ===
    img = _boost_subject_saturation(img, mask=subject_mask)

    # === Step 4: Advanced effects ===
    if "color pop" in style_notes or "selective color" in style_notes:
        img = _selective_color_pop(img, mask=subject_mask)

    if "face glow" in style_notes or "dramatic" in style_notes:
        img = _add_face_glow(img, intensity=0.4 if "dramatic" in style_notes else 0.25)

    if "boost brightness" in style_notes or "brighten" in style_notes:
        img = ImageEnhance.Brightness(img).enhance(1.12)

    if "no vignette" not in style_notes:
        vignette_intensity = 0.35 if "add vignette" in style_notes or "dramatic" in style_notes else 0.18
        img = _add_vignette(img, intensity=vignette_intensity)

    # === Step 5: Arrow overlay (YouTube-style attention arrow) ===
    arrow_config = concept.get("arrow_overlay")
    if isinstance(arrow_config, dict) and arrow_config.get("enabled"):
        img = _add_arrow_overlay(
            img,
            color=arrow_config.get("color", "#FF3333"),
            style=arrow_config.get("style", "straight"),
            target=arrow_config.get("target", "face"),
            subject_mask=subject_mask,
        )

    # === Step 6: Emoji overlays ===
    emoji_specs = concept.get("emoji_overlays")
    if isinstance(emoji_specs, list) and emoji_specs:
        face_bboxes_for_emoji = None
        try:
            faces = detect_faces(np.array(img))
            if faces:
                face_bboxes_for_emoji = [f.bbox for f in faces]
        except Exception:
            pass
        img = _add_emoji_overlays(img, emoji_specs, face_bboxes=face_bboxes_for_emoji)

    # === Step 7: Text overlay (multi-layer compositing) ===
    text = concept.get("text_overlay")
    quote_style = concept.get("quote_style", False)
    if text and text.strip():
        display_text = text.upper()
        if quote_style:
            display_text = f"\u201C{display_text}\u201D"

        text_style = "modern"
        if "text box" in style_notes or "box" in style_notes:
            text_style = "box"
        elif "glow" in style_notes:
            text_style = "glow"
        elif "classic" in style_notes:
            text_style = "classic"

        text_hierarchy = (concept.get("text_hierarchy") or "large").lower()
        font_style = (concept.get("font_style") or "condensed").lower()
        if font_style not in FONT_STYLES:
            font_style = "condensed"

        # Quote style: prefer elegant font for editorial feel
        if quote_style and font_style in ("condensed", "heavy", "classic"):
            font_style = "elegant"

        img = _add_text_overlay(
            img,
            text=display_text,
            position=concept.get("text_position") or "bottom-left",
            text_color=concept.get("text_color") or "#FFFFFF",
            stroke_color=concept.get("text_stroke_color") or "#000000",
            style=text_style,
            hierarchy=text_hierarchy,
            font_style=font_style,
            highlight_word=concept.get("highlight_word"),
            highlight_color=concept.get("highlight_color"),
        )

    # === Step 8: Colored border (popular in modern thumbnails) ===
    if "border" in style_notes or "frame" in style_notes:
        border_color = concept.get("text_color") or concept.get("highlight_color") or "#FFD700"
        img = _add_colored_border(img, color=border_color, width=14)

    # === Step 9: Final polish (single gentle sharpen, no contrast stack) ===
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))

    img.save(str(output_path), "JPEG", quality=100, optimize=False, subsampling=0)
    logger.info(f"Generated thumbnail: {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Background blur / subject isolation
# ---------------------------------------------------------------------------

def _apply_background_blur(
    img: Image.Image, blur_radius: int = 45, darken: float = 0.60,
    mask: np.ndarray | None = None,
) -> Image.Image:
    """
    Isolate subject from background using MediaPipe segmentation.
    Blurs and/or darkens the background while keeping the subject sharp.
    """
    if mask is None:
        img_rgb = np.array(img)
        mask = create_subject_mask(img_rgb)

    # If mask is essentially empty, skip (no person detected)
    if mask.max() < 0.1:
        return img

    # Create background layer
    if blur_radius > 0:
        bg = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    else:
        bg = img.copy()

    # Darken background
    if darken < 1.0:
        bg = ImageEnhance.Brightness(bg).enhance(darken)

    # Composite: sharp subject over blurred/darkened background
    bg_arr = np.array(bg).astype(np.float32)
    fg_arr = np.array(img).astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    result = fg_arr * mask_3ch + bg_arr * (1.0 - mask_3ch)
    return Image.fromarray(result.astype(np.uint8))


# ---------------------------------------------------------------------------
# Smart crop (MediaPipe-based)
# ---------------------------------------------------------------------------

def _smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Intelligent crop that focuses on faces if present, otherwise center crop.
    Uses MediaPipe for accurate face detection including side profiles.

    Goal: head + shoulders framing (face = ~20-35% of thumbnail height).
    NEVER zoom so tight that the eyes are cropped out.
    """
    src_w, src_h = img.size
    target_ratio = target_w / target_h

    face_bbox = None
    face_center = None

    try:
        img_rgb = np.array(img)
        faces = detect_faces(img_rgb)

        if faces:
            face = faces[0]  # largest face
            fx, fy, fw, fh = face.bbox
            face_bbox = (fx, fy, fw, fh)
            # Use a point above the face center (between eyes) as the anchor
            # so the crop shows forehead/hair, not chin
            face_center = (fx + fw // 2, fy + int(fh * 0.35))
    except Exception as e:
        logger.debug(f"Face detection in smart_crop failed: {e}")

    # Gentle zoom: target face height = ~30% of crop height.
    # Only zoom if the face is small; never crop tighter than 1.5x.
    zoom_factor = 1.0
    if face_bbox:
        _, _, _, fh = face_bbox
        face_height_ratio = fh / src_h
        # We want face ~30% of final height. Calculate needed zoom.
        if face_height_ratio < 0.10:
            zoom_factor = min(1.5, 0.30 / face_height_ratio)
        elif face_height_ratio < 0.18:
            zoom_factor = min(1.35, 0.28 / face_height_ratio)
        elif face_height_ratio < 0.25:
            zoom_factor = 1.15
        # face already 25%+ of frame — no zoom needed

    # Apply zoom by cropping before the aspect-ratio crop
    if zoom_factor > 1.0:
        zoom_w = int(src_w / zoom_factor)
        zoom_h = int(src_h / zoom_factor)

        if face_center:
            cx, cy = face_center
            left = max(0, min(src_w - zoom_w, cx - zoom_w // 2))
            top = max(0, min(src_h - zoom_h, cy - zoom_h // 2))
        else:
            left = (src_w - zoom_w) // 2
            top = (src_h - zoom_h) // 2

        img = img.crop((left, top, left + zoom_w, top + zoom_h))
        src_w, src_h = img.size

        if face_center:
            face_center = (face_center[0] - left, face_center[1] - top)

    # Aspect ratio crop
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        if face_center:
            cx = face_center[0]
            left = max(0, min(src_w - new_w, cx - new_w // 2))
        else:
            left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        if face_center:
            cy = face_center[1]
            # Place face in upper-third area of the frame
            ideal_position = int(new_h * 0.35)
            top = max(0, min(src_h - new_h, cy - ideal_position))
        else:
            top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Color grading
# ---------------------------------------------------------------------------

def _apply_color_grade(img: Image.Image, grade_type: str = "authentic") -> Image.Image:
    """
    Apply professional color grading.

    Presets:
    - authentic: Natural but punchy (default)
    - vibrant: Middle-ground between authentic and mrbeast
    - clean: Minimal, Emma Chamberlain style
    - warm: Inviting, cozy feel
    - dark: Dark mode optimized with subtle accents
    - mrbeast: Ultra-saturated (challenge/viral content)
    """
    if grade_type == "mrbeast":
        img = ImageEnhance.Color(img).enhance(1.45)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Brightness(img).enhance(1.15)

    elif grade_type == "vibrant":
        img = ImageEnhance.Color(img).enhance(1.40)
        img = ImageEnhance.Contrast(img).enhance(1.30)
        img = _apply_s_curve_contrast(img, strength=0.30)
        img = ImageEnhance.Brightness(img).enhance(1.10)
        img = ImageEnhance.Sharpness(img).enhance(1.2)

    elif grade_type == "dark":
        img_arr = np.array(img)
        img_arr = np.power(img_arr / 255.0, 1.2) * 255
        img = Image.fromarray(img_arr.astype('uint8'))
        img = ImageEnhance.Color(img).enhance(1.35)
        img = ImageEnhance.Contrast(img).enhance(1.3)

    elif grade_type == "warm":
        img = ImageEnhance.Color(img).enhance(1.15)
        img = ImageEnhance.Brightness(img).enhance(1.08)
        img_arr = np.array(img)
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] * 1.08, 0, 255)  # Red
        img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] * 1.04, 0, 255)  # Green
        img = Image.fromarray(img_arr.astype('uint8'))
        img = ImageEnhance.Contrast(img).enhance(1.15)

    elif grade_type == "cinematic":
        # Teal shadows + warm highlights (Hollywood look) — stronger version
        img_arr = np.array(img).astype(np.float32)
        shadows = img_arr / 255.0
        shadow_mask = 1.0 - shadows
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] - shadow_mask[:, :, 0] * 18, 0, 255)  # less red in shadows
        img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] + shadow_mask[:, :, 2] * 22, 0, 255)  # more blue in shadows
        highlight_mask = shadows
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + highlight_mask[:, :, 0] * 14, 0, 255)  # warm highlights
        img = Image.fromarray(img_arr.astype(np.uint8))
        img = ImageEnhance.Contrast(img).enhance(1.30)
        img = ImageEnhance.Color(img).enhance(1.20)
        img = ImageEnhance.Sharpness(img).enhance(1.15)

    elif grade_type == "clean":
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.03)

    else:  # "authentic" — punchy YouTube-grade enhancement
        img = ImageEnhance.Color(img).enhance(1.30)
        img = ImageEnhance.Contrast(img).enhance(1.25)
        img = _apply_s_curve_contrast(img, strength=0.25)
        img = ImageEnhance.Brightness(img).enhance(1.06)
        img = ImageEnhance.Sharpness(img).enhance(1.15)

    return img


def _apply_s_curve_contrast(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """
    Apply S-curve midtone contrast (like Lightroom tone curve).
    Lifts highlights and deepens shadows without clipping.
    strength: 0.0 = no effect, 1.0 = strong S-curve.
    """
    img_arr = np.array(img).astype(np.float32) / 255.0

    # S-curve: apply sigmoid-like remapping centered at 0.5
    # Using a simple polynomial S-curve: output = x + strength * x * (1-x) * (2*x - 1) * 4
    x = img_arr
    curve = x + strength * x * (1.0 - x) * (2.0 * x - 1.0) * 4.0
    curve = np.clip(curve, 0.0, 1.0)

    return Image.fromarray((curve * 255).astype(np.uint8))


def _boost_subject_saturation(
    img: Image.Image, boost: float = 1.15, mask: np.ndarray | None = None,
) -> Image.Image:
    """
    Boost saturation on the subject (person) more than the background.
    Uses cached or freshly computed MediaPipe segmentation mask.
    """
    if mask is None:
        img_rgb = np.array(img)
        mask = create_subject_mask(img_rgb)

    if mask.max() < 0.1:
        return img

    # Create boosted-saturation version
    boosted = ImageEnhance.Color(img).enhance(boost)
    boosted_arr = np.array(boosted).astype(np.float32)
    original_arr = np.array(img).astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Blend: boosted in subject region, original in background
    result = boosted_arr * mask_3ch + original_arr * (1.0 - mask_3ch)
    return Image.fromarray(result.astype(np.uint8))


# ---------------------------------------------------------------------------
# Selective color pop and face glow (updated to use MediaPipe)
# ---------------------------------------------------------------------------

def _selective_color_pop(
    img: Image.Image, mask: np.ndarray | None = None,
) -> Image.Image:
    """
    Desaturate background while keeping subject saturated.
    Uses cached or freshly computed MediaPipe segmentation mask.
    """
    if mask is None:
        img_rgb = np.array(img)
        mask = create_subject_mask(img_rgb)

    if mask.max() < 0.1:
        return img

    mask_3ch = np.stack([mask] * 3, axis=-1)

    img_desat = img.convert('L').convert('RGB')
    img_desat_arr = np.array(img_desat).astype(np.float32)
    img_arr = np.array(img).astype(np.float32)

    result = (
        img_arr * mask_3ch +
        img_arr * (1 - mask_3ch) * 0.5 +
        img_desat_arr * (1 - mask_3ch) * 0.5
    )

    return Image.fromarray(result.astype(np.uint8))


def _add_face_glow(img: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Add subtle glow/rim light effect to faces."""
    w, h = img.size
    try:
        img_rgb = np.array(img)
        faces = detect_faces(img_rgb)
    except Exception:
        return img

    if not faces:
        return img

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for face in faces:
        fx, fy, fw, fh = face.bbox
        for offset in range(20, 0, -2):
            alpha = int(intensity * 255 * (20 - offset) / 20)
            x1 = fx - offset
            y1 = fy - offset
            x2 = fx + fw + offset
            y2 = fy + fh + offset
            draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255, alpha // 2))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))

    img_rgba = img.convert('RGBA')
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert('RGB')


# ---------------------------------------------------------------------------
# Text rendering overhaul (multi-layer compositing)
# ---------------------------------------------------------------------------

def _lighten_color(hex_color: str, factor: float = 0.4) -> str:
    """
    Auto-compute a lighter shade of a hex color for gradient top.
    factor: 0.0 = same color, 1.0 = white.
    """
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except (ValueError, IndexError):
        return "#FFFFFF"

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def _render_gradient_text(
    size: tuple[int, int],
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    top_color: str,
    bottom_color: str,
    tracking: int = 0,
) -> Image.Image:
    """
    Render text with a vertical color gradient through the letterforms.
    Gradient spans only the text bounding box so the color change is visible.
    Returns an RGBA image with the gradient text.
    """
    w, h = size

    # Create text mask (white text on black) — use tracked rendering to match other layers
    mask_img = Image.new("L", size, 0)
    mask_draw = ImageDraw.Draw(mask_img)
    _draw_tracked_text(mask_draw, position, text, font, tracking, fill=255)

    # Get the actual text bounding box for tight gradient
    # Use textbbox as approximation (close enough for gradient region)
    text_bbox = mask_draw.textbbox(position, text, font=font)
    text_top = max(0, text_bbox[1])
    text_bottom = min(h, text_bbox[3])
    text_h = max(1, text_bottom - text_top)

    try:
        tr = int(top_color[1:3], 16)
        tg = int(top_color[3:5], 16)
        tb = int(top_color[5:7], 16)
        br = int(bottom_color[1:3], 16)
        bg_ = int(bottom_color[3:5], 16)
        bb = int(bottom_color[5:7], 16)
    except (ValueError, IndexError):
        tr, tg, tb = 255, 255, 255
        br, bg_, bb = 200, 200, 200

    # Build gradient array with numpy (fast) — gradient only across text region
    gradient_arr = np.zeros((h, w, 3), dtype=np.uint8)
    t = np.linspace(0, 1, text_h).reshape(-1, 1)
    r_vals = np.clip(tr + (br - tr) * t, 0, 255).astype(np.uint8)
    g_vals = np.clip(tg + (bg_ - tg) * t, 0, 255).astype(np.uint8)
    b_vals = np.clip(tb + (bb - tb) * t, 0, 255).astype(np.uint8)
    gradient_arr[text_top:text_bottom, :, 0] = r_vals
    gradient_arr[text_top:text_bottom, :, 1] = g_vals
    gradient_arr[text_top:text_bottom, :, 2] = b_vals
    # Fill above/below text region with top/bottom colors
    if text_top > 0:
        gradient_arr[:text_top, :] = [tr, tg, tb]
    if text_bottom < h:
        gradient_arr[text_bottom:, :] = [br, bg_, bb]

    gradient = Image.fromarray(gradient_arr)

    # Apply text mask to gradient
    gradient_rgba = gradient.convert("RGBA")
    rc, gc, bc, _ = gradient_rgba.split()
    gradient_rgba = Image.merge("RGBA", (rc, gc, bc, mask_img))

    return gradient_rgba


def _measure_tracked_text(
    draw: ImageDraw.Draw, text: str, font, tracking: int = 0,
) -> tuple[int, int]:
    """Measure text width/height accounting for letter tracking."""
    if tracking <= 0:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    # With tracking: use advance width (getlength) so spaces are measured correctly
    total_w = 0
    max_h = 0
    for i, ch in enumerate(text):
        advance = font.getlength(ch)
        bbox = draw.textbbox((0, 0), ch, font=font)
        ch_h = bbox[3] - bbox[1]
        total_w += advance
        max_h = max(max_h, ch_h)
        if i < len(text) - 1:
            total_w += tracking
    return int(total_w), max_h


def _draw_tracked_text(
    draw: ImageDraw.Draw,
    position: tuple[int, int],
    text: str,
    font,
    tracking: int = 0,
    **kwargs,
):
    """Draw text with optional letter tracking (extra space between chars)."""
    if tracking <= 0:
        draw.text(position, text, font=font, **kwargs)
        return
    x, y = position
    for i, ch in enumerate(text):
        draw.text((x, y), ch, font=font, **kwargs)
        # Use advance width (getlength) so spaces advance correctly
        advance = font.getlength(ch)
        x += advance + tracking


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.Draw, tracking: int = 0) -> list:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        tw, _ = _measure_tracked_text(draw, test_line, font, tracking)
        if tw <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _get_face_regions(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Get face bounding boxes for face-aware text positioning."""
    try:
        img_rgb = np.array(img)
        faces = detect_faces(img_rgb)
        return [f.bbox for f in faces]
    except Exception:
        return []


def _text_overlaps_face(
    text_bbox: tuple[int, int, int, int],
    face_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Check if a text bounding box overlaps any face region."""
    tx1, ty1, tx2, ty2 = text_bbox
    for fx, fy, fw, fh in face_bboxes:
        # Add padding around face
        pad = int(max(fw, fh) * 0.15)
        fx1, fy1 = fx - pad, fy - pad
        fx2, fy2 = fx + fw + pad, fy + fh + pad
        # Check overlap
        if tx1 < fx2 and tx2 > fx1 and ty1 < fy2 and ty2 > fy1:
            return True
    return False


def _add_text_overlay(
    img: Image.Image,
    text: str,
    position: str,
    text_color: str,
    stroke_color: str,
    style: str = "modern",
    hierarchy: str = "large",
    font_style: str = "condensed",
    highlight_word: str | None = None,
    highlight_color: str | None = None,
    min_stroke_width: int = 0,
    add_shadow: bool = False,
) -> Image.Image:
    """
    Add professional text overlay with multi-layer compositing:
    1. Soft drop shadow (blurred black text at offset)
    2. 3D extrusion (multiple offset renders in dark shade)
    3. Outline (thick stroke)
    4. Gradient fill (top=lighter shade, bottom=specified color)
    5. Inner bevel highlight

    min_stroke_width: if > 0, overrides the computed stroke_width floor
                      (used by contrast auto-fix to guarantee legibility).
    add_shadow: if True, ensures the shadow layer is always rendered
                regardless of style (used by contrast auto-fix).

    Supports font_style selection and highlight_word (one word in accent color).
    Includes face-aware positioning to avoid covering faces.
    """
    w, h = img.size
    canvas = img.convert("RGBA")

    text_len = len(text)

    # Scale font based on text_hierarchy
    hierarchy_scale = {"large": 1.0, "medium": 0.8, "small": 0.65}.get(hierarchy, 1.0)

    # Base sizes by text length
    if text_len <= 4:
        base_size = 190
    elif text_len <= 8:
        base_size = 165
    elif text_len <= 14:
        base_size = 135
    elif text_len <= 20:
        base_size = 110
    else:
        base_size = 90

    # Apply font-aware width factor (narrow fonts go bigger, wide fonts smaller)
    style_info = _get_font_style_info(font_style)
    width_factor = style_info.get("width_factor", 1.0)
    tracking = style_info.get("tracking", 0)

    font_size = int(base_size * hierarchy_scale * width_factor)
    font = _get_font(font_style, font_size)

    # Measure text (with tracking)
    temp_draw = ImageDraw.Draw(canvas)
    margin = 50
    max_text_width = w - (margin * 2)

    # Auto-shrink: thumbnail text (≤5 words) should fit on one line.
    # If it wraps, reduce font_size iteratively until it fits.
    word_count = len(text.split())
    min_font_size = max(48, font_size // 2)
    for _ in range(10):  # max 10 shrink steps
        lines = _wrap_text(text, font, max_text_width, temp_draw, tracking)
        if len(lines) <= 1 or word_count > 5 or font_size <= min_font_size:
            break
        font_size = int(font_size * 0.88)
        font = _get_font(font_style, font_size)

    line_spacing = int(font_size * 0.2)
    line_heights = []
    line_widths = []
    for line in lines:
        lw, lh = _measure_tracked_text(temp_draw, line, font, tracking)
        line_widths.append(lw)
        line_heights.append(lh)

    total_text_h = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0
    max_line_w = max(line_widths) if line_widths else 0

    # --- Face-aware positioning (try all 6 positions) ---
    face_bboxes = _get_face_regions(img)

    all_positions = ["top-left", "top-right", "bottom-left", "bottom-right",
                     "center-left", "center-right"]

    def _opposite_vertical(pos: str) -> str:
        if "top" in pos:
            return pos.replace("top", "bottom")
        if "bottom" in pos:
            return pos.replace("bottom", "top")
        return pos

    def _opposite_horizontal(pos: str) -> str:
        if "left" in pos:
            return pos.replace("left", "right")
        if "right" in pos:
            return pos.replace("right", "left")
        return pos

    # Build priority order: requested -> opposite-vertical -> opposite-horizontal
    # -> opposite-both -> remaining two
    opp_v = _opposite_vertical(position)
    opp_h = _opposite_horizontal(position)
    opp_both = _opposite_horizontal(opp_v)
    priority_order = [position, opp_v, opp_h, opp_both]
    for p in all_positions:
        if p not in priority_order:
            priority_order.append(p)

    def _calc_position(pos: str):
        """Return (block_y, align, text_x1) for a given position string."""
        if "top" in pos:
            by = margin
        elif "bottom" in pos:
            by = h - total_text_h - margin - 20
        else:
            by = (h - total_text_h) // 2

        if "left" in pos:
            al = "left"
            tx1 = margin
        elif "right" in pos:
            al = "right"
            tx1 = w - max_line_w - margin
        else:
            al = "left"
            tx1 = margin

        return by, al, tx1

    chosen_position = position
    reduced_hierarchy = False

    if face_bboxes:
        found_clear = False
        for candidate in priority_order:
            by, al, tx1 = _calc_position(candidate)
            region = (tx1, by, tx1 + max_line_w, by + total_text_h)
            if not _text_overlaps_face(region, face_bboxes):
                chosen_position = candidate
                found_clear = True
                break

        if not found_clear:
            # All positions overlap — fall back to bottom-left with smaller text
            chosen_position = "bottom-left"
            reduced_hierarchy = True

    block_y, align, text_x1 = _calc_position(chosen_position)

    # If all positions overlapped faces, reduce hierarchy one level
    if reduced_hierarchy:
        downgrade = {"large": "medium", "medium": "small", "small": "small"}
        hierarchy = downgrade.get(hierarchy, hierarchy)
        hierarchy_scale_new = {"large": 1.0, "medium": 0.8, "small": 0.65}.get(hierarchy, 1.0)
        font_size = int(base_size * hierarchy_scale_new * width_factor)
        font = _get_font(font_style, font_size)
        # Recompute text measurements with smaller font
        lines = _wrap_text(text, font, max_text_width, temp_draw, tracking)
        line_heights = []
        line_widths = []
        for line in lines:
            lw, lh = _measure_tracked_text(temp_draw, line, font, tracking)
            line_widths.append(lw)
            line_heights.append(lh)
        total_text_h = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0
        max_line_w = max(line_widths) if line_widths else 0
        # Recalculate position with new measurements
        block_y, align, text_x1 = _calc_position(chosen_position)

    # --- Style-specific background ---
    if style == "box":
        img = _add_text_box(img, block_y, total_text_h, max_line_w, position, text_color)
        canvas = img.convert("RGBA")
        text_color_fill = "#FFFFFF"
    else:
        img = _add_text_backing(img, position, total_text_h, margin)
        canvas = img.convert("RGBA")
        text_color_fill = text_color

    # --- Multi-layer text compositing ---
    stroke_width = max(max(8, font_size // 10), min_stroke_width)
    shadow_offset = max(8, font_size // 10)
    shadow_blur = max(6, font_size // 20)
    extrusion_depth = max(5, font_size // 18)
    top_color = _lighten_color(text_color_fill, 0.45)

    # Highlight word setup (for two-tone text)
    hl_word = highlight_word.upper().strip() if highlight_word else None
    hl_color = highlight_color or "#FFD700"

    current_y = block_y
    for i, line in enumerate(lines):
        line_w = line_widths[i]
        line_h = line_heights[i]

        if align == "left":
            x = margin
        elif align == "right":
            x = w - line_w - margin
        else:
            x = (w - line_w) // 2

        if style == "glow":
            # For glow style: optionally prepend shadow when contrast auto-fix requests it
            if add_shadow:
                shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                _draw_tracked_text(
                    ImageDraw.Draw(shadow_layer),
                    (x + shadow_offset, current_y + shadow_offset),
                    line, font, tracking, fill=(0, 0, 0, 200),
                )
                shadow_layer = shadow_layer.filter(
                    ImageFilter.GaussianBlur(radius=shadow_blur)
                )
                canvas = Image.alpha_composite(canvas, shadow_layer)
            draw = ImageDraw.Draw(canvas)
            for offset in range(8, 0, -3):
                alpha_hex = hex(int(200 * (8 - offset) / 8))[2:].zfill(2)
                glow_color = text_color_fill + alpha_hex
                _draw_tracked_text(
                    draw, (x, current_y), line, font, tracking,
                    fill=glow_color,
                    stroke_width=stroke_width + offset, stroke_fill=glow_color,
                )
            _draw_tracked_text(
                draw, (x, current_y), line, font, tracking,
                fill=text_color_fill,
                stroke_width=stroke_width, stroke_fill=stroke_color,
            )
        else:
            # Layer 1: Soft drop shadow
            shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            _draw_tracked_text(
                ImageDraw.Draw(shadow_layer),
                (x + shadow_offset, current_y + shadow_offset),
                line, font, tracking, fill=(0, 0, 0, 200),
            )
            shadow_layer = shadow_layer.filter(
                ImageFilter.GaussianBlur(radius=shadow_blur)
            )
            canvas = Image.alpha_composite(canvas, shadow_layer)

            # Layer 2: 3D extrusion with depth gradient
            # Render back-to-front so closer layers paint over farther ones
            extrusion_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ext_draw = ImageDraw.Draw(extrusion_layer)
            for d in range(extrusion_depth, 0, -1):
                # Gradient from dark (far) to slightly lighter (near)
                shade = min(60, int(40 * (1 - d / max(extrusion_depth, 1))))
                alpha = min(240, 180 + d * 6)
                _draw_tracked_text(
                    ext_draw, (x + d, current_y + d),
                    line, font, tracking, fill=(shade, shade, shade, alpha),
                )
            canvas = Image.alpha_composite(canvas, extrusion_layer)

            # Layer 3: Outline (thick stroke)
            outline_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            _draw_tracked_text(
                ImageDraw.Draw(outline_layer), (x, current_y),
                line, font, tracking,
                fill=(0, 0, 0, 0),
                stroke_width=stroke_width, stroke_fill=stroke_color,
            )
            canvas = Image.alpha_composite(canvas, outline_layer)

            # Layer 4: Gradient fill through letterforms
            gradient_layer = _render_gradient_text(
                (w, h), line, (x, current_y), font,
                top_color=top_color,
                bottom_color=text_color_fill,
                tracking=tracking,
            )
            canvas = Image.alpha_composite(canvas, gradient_layer)

            # Layer 4b: Highlight word overlay (two-tone text)
            if hl_word and any(
                w.strip(string.punctuation) == hl_word or w == hl_word
                for w in line.split()
            ):
                canvas = _render_highlight_word(
                    canvas, line, (x, current_y), font, tracking,
                    hl_word, hl_color, stroke_color, stroke_width,
                )

            # Layer 5: Inner bevel highlight
            bevel_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            _draw_tracked_text(
                ImageDraw.Draw(bevel_layer), (x, current_y - 2),
                line, font, tracking, fill=(255, 255, 255, 70),
            )
            text_mask = Image.new("L", (w, h), 0)
            _draw_tracked_text(
                ImageDraw.Draw(text_mask), (x, current_y),
                line, font, tracking, fill=255,
            )
            bevel_layer.putalpha(
                ImageChops.multiply(bevel_layer.split()[3], text_mask)
            )
            canvas = Image.alpha_composite(canvas, bevel_layer)

        current_y += line_h + line_spacing

    return canvas.convert("RGB")


def _render_highlight_word(
    canvas: Image.Image,
    line: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    tracking: int,
    highlight_word: str,
    highlight_color: str,
    stroke_color: str,
    stroke_width: int,
) -> Image.Image:
    """
    Render one word in a different color within a line of text.
    Finds the word's position and overlays it with the accent color.
    """
    w, h = canvas.size
    x, y = position
    words = line.split()
    temp_draw = ImageDraw.Draw(canvas)

    # Walk through words to find the highlight word's x position
    # Strip punctuation for matching (e.g., "15..." should match "15")
    cursor_x = x
    for word in words:
        word_w, word_h = _measure_tracked_text(temp_draw, word, font, tracking)
        space_w, _ = _measure_tracked_text(temp_draw, " ", font, 0)

        word_clean = word.strip(string.punctuation)
        if word_clean == highlight_word or word == highlight_word:
            # Found it — render accent color over just this word
            hl_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            hl_draw = ImageDraw.Draw(hl_layer)
            # Outline
            _draw_tracked_text(
                hl_draw, (cursor_x, y), word, font, tracking,
                fill=(0, 0, 0, 0),
                stroke_width=stroke_width, stroke_fill=stroke_color,
            )
            # Fill with highlight color
            try:
                r = int(highlight_color[1:3], 16)
                g = int(highlight_color[3:5], 16)
                b = int(highlight_color[5:7], 16)
            except (ValueError, IndexError):
                r, g, b = 255, 215, 0
            _draw_tracked_text(
                hl_draw, (cursor_x, y), word, font, tracking,
                fill=(r, g, b, 255),
            )
            canvas = Image.alpha_composite(canvas, hl_layer)
            break

        cursor_x += word_w + space_w + tracking

    return canvas


# ---------------------------------------------------------------------------
# Text background helpers
# ---------------------------------------------------------------------------

def _add_text_box(
    img: Image.Image,
    text_y: int,
    text_height: int,
    text_width: int,
    position: str,
    accent_color: str,
) -> Image.Image:
    """Add a colored box background behind text. Returns new image."""
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        r = int(accent_color[1:3], 16)
        g = int(accent_color[3:5], 16)
        b = int(accent_color[5:7], 16)
    except Exception:
        r, g, b = 255, 215, 0

    padding = 45
    box_height = text_height + padding * 2

    if "left" in position or "right" in position:
        box_width = text_width + padding * 4
        if "left" in position:
            box_x = 0
        else:
            box_x = w - box_width
    else:
        box_width = w
        box_x = 0

    box_y = text_y - padding

    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height],
        radius=20,
        fill=(r, g, b, 240)
    )

    # White highlight gradient on box (vectorized)
    highlight = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(box_height):
        alpha = int(50 * (1 - i / max(box_height, 1)))
        row = box_y + i
        if 0 <= row < h:
            highlight[row, box_x:min(box_x + box_width, w)] = [255, 255, 255, alpha]
    highlight_img = Image.fromarray(highlight, "RGBA")
    overlay = Image.alpha_composite(overlay, highlight_img)

    img_rgba = img.convert('RGBA')
    return Image.alpha_composite(img_rgba, overlay).convert('RGB')


def _add_text_backing(img: Image.Image, position: str, text_height: int, margin: int) -> Image.Image:
    """Add a semi-transparent gradient behind text area for readability. Returns new image."""
    w, h = img.size

    padding = 40
    grad_height = text_height + margin + padding * 2

    # Build gradient overlay with numpy (faster than per-line draw calls)
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

    if "top" in position:
        end_y = min(grad_height, h)
        for y in range(end_y):
            alpha = int(120 * (1 - y / grad_height))
            overlay_arr[y, :] = [0, 0, 0, alpha]
    else:
        start_y = max(0, h - grad_height)
        for y in range(start_y, h):
            progress = (y - start_y) / max(grad_height, 1)
            alpha = int(140 * progress)
            overlay_arr[y, :] = [0, 0, 0, alpha]

    overlay = Image.fromarray(overlay_arr, "RGBA")
    img_rgba = img.convert("RGBA")
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

def _add_vignette(img: Image.Image, intensity: float = 0.2) -> Image.Image:
    """Add a vignette (darkened edges) effect using numpy vectorized radial gradient."""
    w, h = img.size

    # Work at 1/4 resolution for performance, then upscale
    scale = 4
    sw, sh = w // scale, h // scale
    scx, scy = sw / 2.0, sh / 2.0
    s_max_dist = math.sqrt(scx * scx + scy * scy)

    # Vectorized computation with numpy (replaces nested Python loop)
    y_coords, x_coords = np.ogrid[:sh, :sw]
    dx = x_coords - scx
    dy = y_coords - scy
    dist = np.sqrt(dx * dx + dy * dy)
    normalized = dist / s_max_dist
    falloff = np.where(normalized > 0.4, np.clip((normalized - 0.4) / 0.6, 0, None), 0)
    brightness = np.clip(255 * (1 - intensity * (falloff ** 1.5)), 0, 255).astype(np.uint8)

    vignette_small = Image.fromarray(brightness)
    vignette = vignette_small.resize((w, h), Image.BILINEAR)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=w // 20))

    vignette_rgb = Image.merge("RGB", (vignette, vignette, vignette))
    return ImageChops.multiply(img, vignette_rgb)


# ---------------------------------------------------------------------------
# Colored border
# ---------------------------------------------------------------------------

def _add_colored_border(img: Image.Image, color: str = "#FFD700", width: int = 14) -> Image.Image:
    """Add a colored border/frame around the thumbnail with rounded inner corners."""
    if not color or not isinstance(color, str):
        color = "#FFD700"
    try:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
    except (ValueError, IndexError):
        r, g, b = 255, 215, 0

    w, h = img.size
    bordered = Image.new("RGB", (w, h), (r, g, b))
    inner = img.resize((w - width * 2, h - width * 2), Image.LANCZOS)
    bordered.paste(inner, (width, width))
    return bordered


# ---------------------------------------------------------------------------
# Public API: compose_thumbnail and generate_thumbnails
# ---------------------------------------------------------------------------

def compose_thumbnail(frame_path: str, concept: dict, output_path: str) -> str:
    """
    Compose a 1280x720 JPG thumbnail from a real video frame using Pillow effects.

    This function applies the full Pillow pipeline (background blur, vignette,
    color grading, text overlay) to the REAL frame. No AI image generation occurs.
    A 160x90 preview is saved alongside the main thumbnail for legibility testing.

    Args:
        frame_path: Path to the source video frame JPEG.
        concept: Concept dict from analyze_frames_with_claude() specifying text,
                 positioning, style options etc.
        output_path: Desired output path for the 1280x720 JPEG.

    Returns:
        The output_path string (same as input, but confirmed written).
    """
    import uuid as _uuid

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Use a stable thumbnail_id derived from the output filename so the
    # generate_thumbnail() helper writes to exactly the right place.
    thumb_id = output_path_obj.stem
    job_id_dir = output_path_obj.parent

    # generate_thumbnail() builds the path internally as:
    #   output_dir / job_id / f"{thumbnail_id}.jpg"
    # We need to use the parent of the output_path as output_dir and the
    # directory name of output_path as job_id so the path lines up.
    # Since callers pass an explicit path, we call it directly via PIL
    # to avoid path-construction mismatch.

    # Open and crop with face-aware framing
    img = Image.open(frame_path).convert("RGB")
    img = _smart_crop(img, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    style_notes = (concept.get("style_notes") or "").lower()

    # Cache subject mask once (avoids 2-3 redundant MediaPipe inferences)
    img_rgb = np.array(img)
    subject_mask = create_subject_mask(img_rgb)

    # === Step 1: Background blur / subject isolation ===
    mask_coverage = float(np.mean(subject_mask > 0.5)) if subject_mask is not None else 0.0
    bg_treatment = (concept.get("background_treatment") or "blur").lower()
    if mask_coverage < 0.05:
        bg_treatment = "none"
        logger.debug(f"Skipping background blur: mask coverage {mask_coverage:.1%} too low")
    elif mask_coverage < 0.15:
        if bg_treatment == "blur":
            img = _apply_background_blur(img, blur_radius=25, darken=0.65, mask=subject_mask)
            bg_treatment = "done"

    if bg_treatment == "blur":
        img = _apply_background_blur(img, blur_radius=55, darken=0.45, mask=subject_mask)
    elif bg_treatment == "darken":
        img = _apply_background_blur(img, blur_radius=0, darken=0.35, mask=subject_mask)

    # === Step 2: Color grading ===
    if "mrbeast" in style_notes or "ultra bright" in style_notes:
        img = _apply_color_grade(img, "mrbeast")
    elif "dark mode" in style_notes or "neon" in style_notes:
        img = _apply_color_grade(img, "dark")
    elif "warm" in style_notes:
        img = _apply_color_grade(img, "warm")
    elif "clean" in style_notes or "minimal" in style_notes:
        img = _apply_color_grade(img, "clean")
    elif "vibrant" in style_notes:
        img = _apply_color_grade(img, "vibrant")
    elif "cinematic" in style_notes:
        img = _apply_color_grade(img, "cinematic")
    else:
        img = _apply_color_grade(img, "authentic")

    # === Step 3: Subject saturation boost ===
    img = _boost_subject_saturation(img, mask=subject_mask)

    # === Step 4: Advanced effects ===
    if "color pop" in style_notes or "selective color" in style_notes:
        img = _selective_color_pop(img, mask=subject_mask)

    if "face glow" in style_notes or "dramatic" in style_notes:
        img = _add_face_glow(img, intensity=0.4 if "dramatic" in style_notes else 0.25)

    if "boost brightness" in style_notes or "brighten" in style_notes:
        img = ImageEnhance.Brightness(img).enhance(1.12)

    if "no vignette" not in style_notes:
        vignette_intensity = 0.35 if "add vignette" in style_notes or "dramatic" in style_notes else 0.18
        img = _add_vignette(img, intensity=vignette_intensity)

    # === Step 5: Arrow overlay ===
    arrow_config = concept.get("arrow_overlay")
    if isinstance(arrow_config, dict) and arrow_config.get("enabled"):
        img = _add_arrow_overlay(
            img,
            color=arrow_config.get("color", "#FF3333"),
            style=arrow_config.get("style", "straight"),
            target=arrow_config.get("target", "face"),
            subject_mask=subject_mask,
        )

    # === Step 6: Emoji overlays ===
    emoji_specs = concept.get("emoji_overlays")
    if isinstance(emoji_specs, list) and emoji_specs:
        face_bboxes_for_emoji = None
        try:
            faces = detect_faces(np.array(img))
            if faces:
                face_bboxes_for_emoji = [f.bbox for f in faces]
        except Exception:
            pass
        img = _add_emoji_overlays(img, emoji_specs, face_bboxes=face_bboxes_for_emoji)

    # === Step 6.5: Enforce text contrast (auto-fix if below 4:1 ratio) ===
    try:
        img, concept = enforce_text_contrast(img, concept)
    except Exception as e:
        logger.debug(f"Text contrast check skipped: {e}")

    # === Step 7: Text overlay ===
    text = concept.get("text_overlay")
    quote_style = concept.get("quote_style", False)
    if text and text.strip():
        display_text = text.upper()
        if quote_style:
            display_text = f"\u201C{display_text}\u201D"

        text_style = "modern"
        if "text box" in style_notes or "box" in style_notes:
            text_style = "box"
        elif "glow" in style_notes:
            text_style = "glow"
        elif "classic" in style_notes:
            text_style = "classic"

        text_hierarchy = (concept.get("text_hierarchy") or "large").lower()
        font_style = (concept.get("font_style") or "condensed").lower()
        if font_style not in FONT_STYLES:
            font_style = "condensed"

        if quote_style and font_style in ("condensed", "heavy", "classic"):
            font_style = "elegant"

        img = _add_text_overlay(
            img,
            text=display_text,
            position=concept.get("text_position") or "bottom-left",
            text_color=concept.get("text_color") or "#FFFFFF",
            stroke_color=concept.get("text_stroke_color") or "#000000",
            style=text_style,
            hierarchy=text_hierarchy,
            font_style=font_style,
            highlight_word=concept.get("highlight_word"),
            highlight_color=concept.get("highlight_color"),
            min_stroke_width=concept.get("text_stroke_width", 0),
            add_shadow=bool(concept.get("text_shadow", False)),
        )

    # === Step 8: Colored border ===
    if "border" in style_notes or "frame" in style_notes:
        border_color = concept.get("text_color") or concept.get("highlight_color") or "#FFD700"
        img = _add_colored_border(img, color=border_color, width=14)

    # === Step 9: Final polish ===
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))

    # === Step 10: Quality gate — squint test ===
    try:
        squint_result = check_squint_test(img, subject_mask)
        if not squint_result["passed"]:
            logger.warning(
                f"compose_thumbnail: squint test FAILED — subject coverage "
                f"{squint_result['subject_coverage']:.0%} at 160px "
                f"(need 15%+). Thumbnail may not be identifiable at small sizes."
            )
    except Exception as e:
        logger.debug(f"Squint test skipped: {e}")

    # Save main 1280x720 thumbnail
    img.save(str(output_path_obj), "JPEG", quality=100, optimize=False, subsampling=0)
    logger.info(f"compose_thumbnail: saved {output_path_obj}")

    # Save 160x90 preview for legibility testing
    preview_stem = output_path_obj.stem + "_preview"
    preview_path = output_path_obj.parent / f"{preview_stem}.jpg"
    preview = img.resize((160, 90), Image.LANCZOS)
    preview.save(str(preview_path), "JPEG", quality=85)
    logger.debug(f"compose_thumbnail: saved preview {preview_path}")

    return str(output_path_obj)


def generate_thumbnails(
    frames: list[dict],
    transcript: dict | str,
    job_id: str,
    enhance: bool = False,
    video_title: str = "",
) -> list[dict]:
    """
    Orchestrate the full thumbnail generation pipeline.

    1. Calls analyze_frames_with_claude() to get 3-5 concept dicts.
    2. For each concept, calls compose_thumbnail() to produce a 1280x720 Pillow JPG.
    3. If enhance=True and GEMINI_API_KEY is set, passes each Pillow output to
       enhance_thumbnail_with_gemini() for optional background stylization.

    The Gemini step is a best-effort post-processor. Failures degrade gracefully
    to the Pillow-only output without failing the job.

    Args:
        frames: List of frame dicts from extract_candidate_frames().
        transcript: Transcript dict (with "text" key) or raw string.
        job_id: Job UUID used to scope output directory.
        enhance: If True and GEMINI_API_KEY is set, apply Gemini enhancement.
        video_title: Optional video title for context.

    Returns:
        List of result dicts, each with: file_path, frame_index, concept,
        generation_type ("pillow" or "gemini_enhanced"), thumb_id.
    """
    import uuid as _uuid
    from backend.config import settings

    # Extract transcript text
    if isinstance(transcript, dict):
        transcript_text = transcript.get("text", "")
    else:
        transcript_text = str(transcript)

    # Pick top frames by score and generate concepts locally (no Claude API call)
    logger.info(f"[{job_id}] Selecting top frames by score (local, no API call)...")
    concepts = _generate_local_concepts(frames, video_title=video_title)

    if not concepts:
        logger.error(f"[{job_id}] No frames available for thumbnail generation")
        return []

    logger.info(f"[{job_id}] Generated {len(concepts)} concept(s) locally. Compositing with Pillow...")

    results = []
    output_dir = THUMBNAILS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for concept in concepts:
        frame_idx = concept.get("frame_index", 0)

        # Clamp to valid frame range
        if frame_idx >= len(frames):
            frame_idx = 0
        frame_path = frames[frame_idx]["file_path"] if frames else None
        if not frame_path or not Path(frame_path).exists():
            logger.warning(f"[{job_id}] Frame file missing for index {frame_idx}, skipping concept")
            continue

        thumb_id = str(_uuid.uuid4())[:8]
        output_path = str(output_dir / f"{thumb_id}.jpg")

        try:
            pillow_path = compose_thumbnail(frame_path, concept, output_path)
        except Exception as exc:
            logger.error(f"[{job_id}] compose_thumbnail failed for concept {thumb_id}: {exc}")
            continue

        generation_type = "pillow"

        # Optional Gemini enhancement post-processing
        if enhance and settings.gemini_api_key:
            try:
                from backend.services.gemini_thumbnail import enhance_thumbnail_with_gemini
                final_path = enhance_thumbnail_with_gemini(pillow_path, concept, video_title)
                generation_type = "gemini_enhanced"
            except Exception as exc:
                logger.warning(
                    f"[{job_id}] Gemini enhancement failed for {thumb_id}, "
                    f"using Pillow output: {exc}"
                )
                final_path = pillow_path
        else:
            final_path = pillow_path

        results.append({
            "thumb_id": thumb_id,
            "frame_index": frame_idx,
            "file_path": final_path,
            "concept": concept,
            "generation_type": generation_type,
        })

    logger.info(f"[{job_id}] Generated {len(results)}/{len(concepts)} thumbnails")
    return results
