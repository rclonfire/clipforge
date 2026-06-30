"""
Gemini thumbnail enhancement service.

Architecture (THMB-03):
- PRIMARY: enhance_thumbnail_with_gemini() — post-processing step that receives
  the Pillow-composited 1280x720 thumbnail and sends it to Gemini for optional
  background stylization. Gemini enhances the REAL frame — it does not generate
  thumbnails from scratch. Uses gemini-2.5-flash-image (production model).

- LEGACY (preserved for potential future use): generate_gemini_thumbnails_standalone()
  — the original approach that generates thumbnails from scratch using headshots.
  Retained for reference but NOT used in the main pipeline.

Usage in pipeline (thumbnail_generator.py):
    pillow_path = compose_thumbnail(frame_path, concept, output_path)
    if enhance and settings.gemini_api_key:
        from backend.services.gemini_thumbnail import enhance_thumbnail_with_gemini
        final_path = enhance_thumbnail_with_gemini(pillow_path, concept, video_title)
    else:
        final_path = pillow_path
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from backend.config import GEMINI_API_KEY, GEMINI_HEADSHOTS_DIR, THUMBNAILS_DIR

# Enable AVIF support for YouTube thumbnails
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)


def get_headshots() -> list[Path]:
    """Load all headshot image paths from the assets folder."""
    headshots_dir = Path(GEMINI_HEADSHOTS_DIR)
    if not headshots_dir.exists():
        logger.warning(f"Headshots directory not found: {headshots_dir}")
        return []

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    headshots = sorted(
        p for p in headshots_dir.iterdir()
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )

    if not headshots:
        logger.warning(f"No headshot images found in {headshots_dir}")

    return headshots


def _resize_if_needed(img: Image.Image, max_edge: int = 2048) -> Image.Image:
    """Resize image if larger than max_edge on any side."""
    w, h = img.size
    if max(w, h) > max_edge:
        ratio = max_edge / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size, Image.LANCZOS)
    return img


def concept_to_gemini_prompt(concept: dict, video_title: str) -> str:
    """
    Convert a Claude concept dict into a detailed Gemini prompt.

    The prompt follows the SKILL.md template: person on right ~40%,
    dark cinematic background, visual elements on left, bold text, 16:9.
    """
    text_overlay = concept.get("text_overlay") or ""
    style_notes = (concept.get("style_notes") or "").lower()
    text_position = concept.get("text_position") or "bottom-left"
    reasoning = concept.get("reasoning") or ""

    # Determine person emotion from style notes and reasoning
    emotion = "confident"
    if any(w in style_notes for w in ["warm", "authentic", "friendly"]):
        emotion = "warm and approachable, with a slight smile"
    elif any(w in style_notes for w in ["vibrant", "energetic", "bold"]):
        emotion = "excited and energetic"
    elif any(w in style_notes for w in ["cinematic", "dramatic", "dark"]):
        emotion = "serious and determined"
    elif any(w in style_notes for w in ["curious", "surprised"]):
        emotion = "curious and intrigued"

    # Determine color direction
    color_direction = "Dark, moody"
    if any(w in style_notes for w in ["warm", "gold", "orange"]):
        color_direction = "Dark with warm amber/gold accent"
    elif any(w in style_notes for w in ["cool", "blue", "cyan"]):
        color_direction = "Dark with cool blue/cyan accent"
    elif any(w in style_notes for w in ["vibrant", "red", "bold"]):
        color_direction = "Dark with bold, saturated red/magenta accent"
    elif any(w in style_notes for w in ["clean", "minimal"]):
        color_direction = "Dark with high-contrast white/minimal accent"

    # Map text_position to prompt description
    position_map = {
        "top-left": "in the upper-left area",
        "top-right": "in the upper-right area",
        "top-center": "centered at the top",
        "center-left": "centered on the left side",
        "center-right": "centered on the right side",
        "center": "centered in the frame",
        "bottom-left": "in the lower-left area",
        "bottom-right": "in the lower-right area (keep above the timestamp zone)",
        "bottom-center": "centered at the bottom",
    }
    text_placement = position_map.get(text_position, "in the lower-left area")

    # Build the prompt
    prompt_parts = [
        f"A professional YouTube video thumbnail in 16:9 aspect ratio for a video titled \"{video_title}\".",
        "",
        "ATTACHED IMAGES:",
        "- Image 1+ (headshot): Reference photos of the person to include. Use their exact likeness, skin tone, and features.",
        "- Final image: A video frame showing the setting/mood of the video. Use this for background inspiration.",
        "",
        "PERSON:",
        f"Use the exact likeness from the headshot reference photo(s). Place them on the right side of the frame, taking up approximately 40% of the width. Show them from the waist up or shoulders up. They should have dramatic, natural lighting on their face with the dark background behind them. Their expression is {emotion}. Use shadows/shading behind the person to create visual separation from the background.",
    ]

    # Background
    prompt_parts.extend([
        "",
        "BACKGROUND:",
        f"Dark, moody, cinematic background — NOT a solid black void. Use a darkened real-world scene inspired by the attached video frame. The scene should feel like dramatic night photography with real environmental detail, texture, and depth. {color_direction} color tones. No glow effects. No bright or white backgrounds, and never a flat solid-color void.",
    ])

    # Visual elements
    prompt_parts.extend([
        "",
        "VISUAL ELEMENTS (left side):",
        f"Based on the video topic \"{video_title}\", include relevant visual elements on the left side of the frame — these could be icons, graphics, or imagery that represent the video's core subject. Keep to a maximum of 3 distinct elements. Elements should be large enough to be visible at small sizes.",
    ])

    # Text
    if text_overlay:
        prompt_parts.extend([
            "",
            "TEXT:",
            f"\"{text_overlay}\" in bold, large, white text. Placed {text_placement}. Clean, heavy, modern sans-serif font. High contrast against the dark background. Must be clearly readable at small sizes. Do NOT overlap text with the person's face.",
        ])
    else:
        prompt_parts.extend([
            "",
            "TEXT:",
            "No text overlay — let the visual composition and the person's expression tell the story.",
        ])

    # Style
    prompt_parts.extend([
        "",
        "STYLE:",
        "Professional, high-contrast, clean design. Similar to top YouTube tech/business channel thumbnails. Dramatic lighting on the person. Subtle depth with layered elements. Polished and modern — not cluttered. Bottom-right corner should be kept clear (YouTube timestamp overlay covers that area).",
    ])

    # Additional style context
    if "vignette" in style_notes:
        prompt_parts.append("Apply a subtle vignette effect around the edges.")

    return "\n".join(prompt_parts)


def generate_thumbnail(
    prompt: str,
    headshot_paths: list[Path],
    frame_path: str | None,
    output_path: Path,
) -> str:
    """
    Call Gemini API with headshots + frame + prompt to generate a thumbnail.

    Returns the output file path as a string.
    Raises an error if Gemini returns no image.
    """
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Cannot generate thumbnails. "
            "Add GEMINI_API_KEY to your .env file."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build contents: prompt text + headshot images + frame image
    contents: list = [prompt]

    # Add headshot(s) — limit to 2 to save context
    for hp in headshot_paths[:2]:
        try:
            img = _resize_if_needed(Image.open(hp))
            contents.append(img)
        except Exception as e:
            logger.warning(f"Could not load headshot {hp}: {e}")

    # Add frame as reference for scene/mood
    if frame_path:
        try:
            frame_img = _resize_if_needed(Image.open(frame_path))
            contents.append(frame_img)
        except Exception as e:
            logger.warning(f"Could not load frame {frame_path}: {e}")

    # Generate with retry for rate limits
    import time

    logger.info(f"Calling Gemini with {len(contents) - 1} images, prompt length {len(prompt)}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="2K",
                    ),
                ),
            )
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    logger.warning(f"Gemini rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            raise

    # Extract and save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_saved = False
    text_response = ""

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            img = part.as_image()
            img.save(str(output_path))
            image_saved = True
            logger.info(f"Gemini thumbnail saved to: {output_path}")
        elif hasattr(part, "text") and part.text is not None:
            text_response += part.text

    if text_response:
        logger.info(f"Gemini notes: {text_response[:200]}")

    if not image_saved:
        error_msg = f"Gemini returned no image. Model response: {text_response[:500]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return str(output_path)


def generate_gemini_thumbnails_standalone(
    concepts: list[dict],
    frames: list[dict],
    video_title: str,
    job_id: str,
) -> list[dict]:
    """
    [LEGACY - preserved for potential future use]

    Generate Gemini thumbnails from scratch for each concept using headshots.

    This was the original generation approach. The current pipeline uses
    enhance_thumbnail_with_gemini() instead, which post-processes Pillow output
    rather than generating from scratch.

    For each concept:
    1. Build a Gemini prompt from the concept data
    2. Pick the matching frame as a reference image
    3. Send headshots + frame + prompt to Gemini
    4. Save the generated image

    Returns a list of dicts with thumbnail info (one per successful generation).
    Failed concepts are logged and skipped (partial results OK).
    """
    headshots = get_headshots()
    if not headshots:
        raise RuntimeError(
            "No headshot images found. Add headshot photos to "
            f"{GEMINI_HEADSHOTS_DIR}/ before generating thumbnails."
        )

    results = []

    def _generate_one(concept: dict, index: int) -> dict | None:
        thumb_id = str(uuid.uuid4())[:8]
        frame_idx = concept.get("frame_index", 0)

        # Clamp frame_index to valid range
        if frame_idx >= len(frames):
            frame_idx = 0

        frame_path = frames[frame_idx]["file_path"] if frames else None

        # Build prompt from concept
        prompt = concept_to_gemini_prompt(concept, video_title)

        # Output path
        output_path = THUMBNAILS_DIR / job_id / f"{thumb_id}.png"

        try:
            file_path = generate_thumbnail(
                prompt=prompt,
                headshot_paths=headshots,
                frame_path=frame_path,
                output_path=output_path,
            )
            return {
                "thumb_id": thumb_id,
                "frame_index": frame_idx,
                "file_path": file_path,
                "concept": concept,
                "generation_type": "gemini",
            }
        except Exception as e:
            logger.error(f"Gemini generation failed for concept {index + 1}: {e}")
            return None

    # Run generations sequentially to respect free-tier API rate limits
    # (bump max_workers to 3 if on a paid Gemini plan)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(_generate_one, concept, i): i
            for i, concept in enumerate(concepts)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"Thumbnail {idx + 1}/{len(concepts)} generated successfully")
                else:
                    logger.warning(f"Thumbnail {idx + 1}/{len(concepts)} skipped (generation failed)")
            except Exception as e:
                logger.error(f"Thumbnail {idx + 1}/{len(concepts)} raised exception: {e}")

    if not results:
        raise RuntimeError(
            "All Gemini thumbnail generations failed. "
            "Check the logs for details and verify your GEMINI_API_KEY is valid."
        )

    logger.info(f"Generated {len(results)}/{len(concepts)} thumbnails successfully")
    return results


# ---------------------------------------------------------------------------
# Primary API: enhance_thumbnail_with_gemini (THMB-03)
# ---------------------------------------------------------------------------

def enhance_thumbnail_with_gemini(
    thumbnail_path: str,
    concept: dict,
    video_title: str,
) -> str:
    """
    Optional post-processing step: enhance a Pillow-composited thumbnail with Gemini.

    This function receives the REAL Pillow output (1280x720 JPG) and sends it to
    Gemini with an enhancement prompt. Gemini improves ONLY the background with
    cinematic lighting and stylization — it does NOT change the person's face or text.

    If Gemini is unavailable (missing API key, API error, or any exception),
    the original Pillow thumbnail is returned unchanged. This ensures graceful
    degradation without failing the job.

    Args:
        thumbnail_path: Path to the Pillow-composited 1280x720 thumbnail JPG.
                        This is the REAL frame composite, not raw video frames.
        concept: Concept dict from analyze_frames_with_claude() for style context.
        video_title: Video title for prompt context.

    Returns:
        Path to the enhanced thumbnail (overwrites thumbnail_path on success),
        or thumbnail_path unchanged if enhancement fails.
    """
    if not GEMINI_API_KEY:
        logger.warning(
            "enhance_thumbnail_with_gemini: GEMINI_API_KEY not set. "
            "Returning Pillow thumbnail unchanged."
        )
        return thumbnail_path

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning(
            "enhance_thumbnail_with_gemini: google-genai package not installed. "
            "Returning Pillow thumbnail unchanged."
        )
        return thumbnail_path

    try:
        style_notes = (concept.get("style_notes") or "").lower()
        text_overlay = concept.get("text_overlay") or ""

        # Build enhancement prompt — natural creative direction for comedy/lifestyle content
        background_direction = "warm, energetic"
        if "warm" in style_notes:
            background_direction = "golden-hour warmth with amber tones and soft light wrapping around the subject"
        elif "cool" in style_notes or "blue" in style_notes:
            background_direction = "cool blue-teal cinematic atmosphere with neon edge lighting"
        elif "dark mode" in style_notes:
            background_direction = "deep black with dramatic rim lighting and a subtle colored accent glow"
        elif "vibrant" in style_notes:
            background_direction = "rich saturated gradient with vivid color pops and studio lighting"
        elif "cinematic" in style_notes:
            background_direction = "Hollywood teal-and-orange grading with shallow depth of field"
        elif "clean" in style_notes or "minimal" in style_notes:
            background_direction = "clean modern gradient, soft and professional like a studio portrait"

        enhancement_prompt = (
            f'This is a YouTube thumbnail for "{video_title}". '
            f"Enhance it to look like it was shot in a professional studio.\n\n"
            f"Give the background a {background_direction} feel — "
            "think cinematic depth with soft bokeh and natural atmosphere, "
            "not a flat digital gradient.\n\n"
            "For the person: warm up skin tones slightly, sharpen the eyes, "
            "and add subtle rim lighting on their edges to separate them from "
            "the background. Keep their face, expression, and features exactly "
            "as they are — just more polished, like professional portrait retouching.\n\n"
            "Boost overall contrast about 20%. Push shadows deeper and highlights brighter "
            "for that punchy YouTube look.\n\n"
            "Keep any text overlays exactly as they are — same words, position, and colors.\n\n"
            "Output at 1280x720."
        )

        # Load and resize the Pillow thumbnail if needed
        pillow_img = _resize_if_needed(Image.open(thumbnail_path))

        client = genai.Client(api_key=GEMINI_API_KEY)
        contents = [enhancement_prompt, pillow_img]

        logger.info(
            f"enhance_thumbnail_with_gemini: sending {thumbnail_path} to Gemini "
            f"for background enhancement (title: {video_title[:50]})"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="2K",
                ),
            ),
        )

        # Extract and save the enhanced image
        image_saved = False
        text_response = ""

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                enhanced_img = part.as_image()
                # Save over the original Pillow output
                enhanced_img.save(str(thumbnail_path), "JPEG", quality=100, optimize=False, subsampling=0)
                image_saved = True
                logger.info(f"enhance_thumbnail_with_gemini: saved enhanced thumbnail to {thumbnail_path}")
            elif hasattr(part, "text") and part.text is not None:
                text_response += part.text

        if text_response:
            logger.debug(f"enhance_thumbnail_with_gemini Gemini notes: {text_response[:200]}")

        if not image_saved:
            logger.warning(
                f"enhance_thumbnail_with_gemini: Gemini returned no image "
                f"(response: {text_response[:200]}). Returning Pillow thumbnail unchanged."
            )

        return thumbnail_path

    except Exception as exc:
        logger.warning(
            f"enhance_thumbnail_with_gemini: Gemini enhancement failed ({exc}). "
            "Returning Pillow thumbnail unchanged (graceful degradation)."
        )
        return thumbnail_path
