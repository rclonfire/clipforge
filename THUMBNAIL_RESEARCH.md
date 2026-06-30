# Thumbnail Enhancement Research

**Date:** 2026-03-22
**Context:** ClipForge v2 currently uses Claude to select best frames, Pillow for compositing, and optionally Gemini for background enhancement. This document covers research into making the output look professional rather than amateur.

---

## 1. What Makes YouTube Thumbnails Look Professional vs Amateur

### The Core Differentiators

Professional thumbnails share five traits that amateur ones lack: **deliberate contrast**, **strategic color**, **face prominence**, **clean text treatment**, and **compositional simplicity**.

### Contrast & Visual Hierarchy

- Target a **4.5:1 contrast ratio** between text and background (WCAG AA standard). 3:1 is acceptable for large/bold text (18pt+).
- The primary subject should fill **40-60% of the frame**. Secondary elements get minimal space.
- Limit designs to **2-3 visual elements maximum**. At mobile sizes (120-160px width), anything more becomes noise.
- The **Squint Test**: if you squint at the thumbnail and can't immediately identify the subject and text, it's too cluttered.
- Target **30-40% negative space** to prevent cramping and accommodate YouTube's UI overlays.

### Color Theory

Best complementary pairs for maximum separation:
- **Blue/Orange** (the Hollywood standard -- teal-and-orange grading)
- **Yellow/Violet**
- **Red/Cyan**

Background saturation should be **reduced to 60-80%** when emphasizing subjects, without removing environmental context entirely. This is the single biggest difference between amateur and professional thumbnails -- amateurs leave background saturation at 100%, which competes with the subject.

### Face Prominence

- **72% of top-performing thumbnails** use exaggerated facial features (analysis of ~1,000 thumbnails).
- Thumbnails with clear, strong emotions boost CTR by **more than 35%**.
- **Gaze direction matters**: position eyes toward text or the promised content. Avoid looking out-of-frame.
- Expression intensity should match actual content tone. Audiences reject exaggerated reactions to mundane content.

### Text Treatment

- **3-5 words maximum**, readable at mobile size (320x180px).
- Sans-serif only. Bold/extra-bold weights mandatory.
- White text with black outline is universally effective.
- Text should **complete the visual, not describe it**. If the face shows shock, don't write "SHOCKED" -- write what caused it ("IT BROKE").

### Critical UI Overlay Zones to Avoid

- **Bottom-right**: Duration badge (always present)
- **Bottom-left**: Watch Later/queue icons
- **Top-right**: Menu dots on hover
- **Bottom 15%**: Progress bar zone

### Composition Frameworks

- **Rule of thirds**: position dominant subjects at intersection points.
- **Layering for depth**: background -> subject -> text overlay creates 3D perception.
- **AIDA model**: Attention (color/contrast) -> Interest (curiosity gap) -> Desire (compelling question) -> Action (mobile readability).

---

## 2. Best Programmatic Approaches (2025-2026)

### 2A. Gemini Image Editing API (Recommended Primary Enhancement)

**Available Models (current as of March 2026):**

| Model | Codename | Price/Image | Best For |
|-------|----------|-------------|----------|
| `gemini-2.5-flash-image` | Nano Banana | ~$0.039 | Speed, high-volume, cost-effective |
| `gemini-3-pro-image-preview` | Nano Banana Pro | ~$0.134 | Best quality, advanced reasoning |
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | ~$0.045 | Newest, fast with thinking |

**Key capability for ClipForge: Image Editing (not generation from scratch)**

You send an existing image + a text prompt describing modifications. The model analyzes the original image's style, lighting, and perspective and makes edits that look natural.

**Working Python code for image editing:**

```python
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="YOUR_KEY")

# Load existing video frame / composited thumbnail
thumbnail = Image.open("composited_thumbnail.jpg")

# Enhancement prompt
prompt = """
Enhance this YouTube thumbnail:
- Increase contrast on the subject's face, make skin tones warm and vibrant
- Darken and blur the background slightly for depth-of-field effect
- Add subtle cinematic color grading (teal shadows, warm highlights)
- Add a subtle rim light / edge glow on the person to separate them from background
- Keep all text overlays exactly as they are
- Keep the person's face, expression, and features pixel-perfect identical
- Output at 1280x720
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-image",  # or gemini-3-pro-image-preview for best quality
    contents=[prompt, thumbnail],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="2K",  # Options: "1K" (default), "2K", "4K"
        ),
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        enhanced = part.as_image()
        enhanced.save("enhanced_thumbnail.jpg", "JPEG", quality=95)
```

**Supported aspect ratios:** 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

**Supported resolutions:** "1K" (default), "2K", "4K" (use uppercase K)

**Key prompting insight**: "Describe the scene, don't just list keywords." Natural language outperforms tag-soup prompting. Act like a Creative Director, not a keyword stuffer.

**Enhancement prompts that work for real photo editing:**
- "Enhance the colors, making them more vibrant and saturated."
- "Increase the contrast for a more dramatic effect."
- "Make the background darker."
- "Warm up the color temperature."
- "Add dramatic studio lighting with rim light separation."
- Use photographic terms: "85mm portrait lens", "shallow depth of field", "golden hour lighting"

**What ClipForge should change:**
The current `enhance_thumbnail_with_gemini()` uses `gemini-2.0-flash-exp-image-generation` which is an older experimental model. Upgrade to `gemini-2.5-flash-image` (production-ready, supports aspect ratio and resolution config) or `gemini-3-pro-image-preview` for best quality.

### 2B. Two-Pass Enhancement Strategy (Recommended Architecture)

Instead of the current single-pass Gemini enhancement, use a **two-pass approach**:

**Pass 1: Local enhancement (free, fast, deterministic)**
Apply OpenCV/Pillow filters to the raw video frame BEFORE compositing:
- CLAHE contrast enhancement on the subject
- Saturation boost (1.2-1.4x)
- Unsharp mask sharpening
- Background darkening/blur via face detection mask
- Cinematic color grading via LUT or curves

**Pass 2: AI enhancement (optional, costs money)**
Send the composited thumbnail to Gemini for:
- Background replacement/stylization
- Rim lighting addition
- Professional color grading refinement
- Overall polish

This way, even without Gemini, thumbnails look significantly better than raw frames.

### 2C. OpenCV Enhancement Pipeline (Free, Local)

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**

```python
import cv2
import numpy as np

def enhance_frame(frame_path: str) -> np.ndarray:
    img = cv2.imread(frame_path)

    # Convert to LAB color space (separates luminance from color)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to luminance channel only
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced
```

**Unsharp Mask Sharpening:**

```python
def sharpen_frame(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Unsharp mask: I_sharp = (1+alpha)*I - alpha*blur"""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return sharpened
```

**Saturation Boost:**

```python
def boost_saturation(img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**Cinematic Teal-Orange Grade:**

```python
def teal_orange_grade(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Apply Hollywood teal-and-orange color grading."""
    result = img.astype(np.float32)

    # Push shadows toward teal, highlights toward orange
    b, g, r = cv2.split(result)

    # Shadows: boost blue+green (teal), reduce red
    shadow_mask = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    shadow_mask = np.clip(1.0 - shadow_mask, 0, 1) ** 2  # stronger in darks

    b += shadow_mask * strength * 40   # blue up in shadows
    g += shadow_mask * strength * 15   # slight green up
    r -= shadow_mask * strength * 20   # red down in shadows

    # Highlights: boost red+green (warm), reduce blue
    highlight_mask = np.clip(shadow_mask * -1 + 1, 0, 1) ** 2
    r += highlight_mask * strength * 30
    g += highlight_mask * strength * 10
    b -= highlight_mask * strength * 15

    result = cv2.merge([
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8),
    ])
    return result
```

### 2D. FFmpeg Filters for Frame Enhancement

FFmpeg can apply cinematic grading to extracted frames before they reach Pillow.

**Key filters:**

```bash
# Boost contrast and saturation on a single frame
ffmpeg -i frame.jpg \
  -vf "eq=contrast=1.3:brightness=0.05:saturation=1.4" \
  enhanced_frame.jpg

# Apply a LUT (Look Up Table) for cinematic grading
ffmpeg -i frame.jpg -i cinematic_lut.png \
  -filter_complex haldclut \
  graded_frame.jpg

# Vignette effect
ffmpeg -i frame.jpg \
  -vf "vignette=PI/4" \
  vignetted_frame.jpg

# Combined cinematic pipeline
ffmpeg -i frame.jpg \
  -vf "eq=contrast=1.2:saturation=1.3,unsharp=5:5:1.0,vignette=PI/5" \
  cinematic_frame.jpg
```

**LUT workflow:**
1. Generate a HALD CLUT: `ffmpeg -f lavfi -i haldclutsrc=8 identity_lut.png`
2. Edit the LUT PNG in any image editor (adjust curves, temperature, color balance)
3. Apply the modified LUT to frames: `ffmpeg -i frame.jpg -i modified_lut.png -filter_complex haldclut graded.jpg`

This is powerful because you can create a "house style" LUT once and apply it consistently to all thumbnails.

### 2E. Python Libraries Beyond Basic Pillow

**PicTex** -- Professional text effects with minimal code:

```python
from pictex import Canvas, Text, Shadow

canvas = (
    Canvas()
    .font_size(120)
    .font_family("Impact")
    .padding(20)
)

# Drop shadow
image = canvas.render(
    Text("IT WORKED")
    .color("white")
    .text_stroke(width=5, color="black")
    .text_shadows(
        Shadow(offset=(6, 6), blur_radius=8, color="#00000080")
    )
)

# Neon glow effect
image = canvas.render(
    Text("NEON")
    .color("#00FFAA")
    .text_shadows(
        Shadow(offset=(0, 0), blur_radius=3, color="#00FFAA80"),
        Shadow(offset=(0, 0), blur_radius=5, color="#00FFAA80"),
        Shadow(offset=(0, 0), blur_radius=10, color="#FFFFFF80"),
    )
)
```

**rembg** -- AI background removal for subject isolation:

```python
from rembg import remove
from PIL import Image

# Remove background from video frame to isolate subject
input_img = Image.open("frame.jpg")
output_img = remove(input_img, model_name="birefnet-general")
# output_img now has transparent background -- composite onto styled bg
```

The `birefnet-general` model is the newest and most accurate for professional results. This enables a workflow where you:
1. Extract the subject from the video frame
2. Apply enhancement only to the subject (sharpening, color correction)
3. Create a separate styled/graded background
4. Composite subject onto background with rim lighting effects

**Pillow's built-in enhancement tools** (already imported in ClipForge but underutilized):

```python
from PIL import ImageEnhance, ImageFilter

# Contrast boost
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.3)  # 1.0 = original, 1.3 = 30% more contrast

# Saturation boost
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(1.4)  # 40% more saturated

# Sharpness
enhancer = ImageEnhance.Sharpness(img)
img = enhancer.enhance(1.5)  # 50% sharper

# Background blur (using face mask)
blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=12))
# Composite: sharp subject over blurred background using mask
result = Image.composite(img, blurred_bg, subject_mask)
```

---

## 3. The "Nano Banana" / Gemini Image Generation Approach

### What is Nano Banana?

"Nano Banana" is the creator community's name for Google's Gemini native image generation models. The current lineup (March 2026):

- **Nano Banana** = `gemini-2.5-flash-image` (production-ready, fast, ~$0.039/image)
- **Nano Banana Pro** = `gemini-3-pro-image-preview` (best quality, advanced reasoning, ~$0.134/image)
- **Nano Banana 2** = `gemini-3.1-flash-image-preview` (newest, includes "thinking" process)

### How Creators Use It for Thumbnails

The primary use case is **generating thumbnail compositions from scratch** using detailed scene descriptions, NOT editing existing photos. However, it also supports editing existing images (which is more relevant to ClipForge).

**Key insight: Nano Banana Pro achieves 94% text rendering accuracy** compared to 60-70% in competing models. It uses multimodal reasoning -- analyzing composition and text placement before rendering pixels.

### Effective Prompt Structure for YouTube Thumbnails

```
A professional YouTube thumbnail in 16:9 aspect ratio, 1280x720.

SUBJECT: [Detailed description of person -- exact appearance, expression, clothing]
placed on the [left/right] third of the frame, medium close-up shot, waist up.
Expression is [specific emotion -- shocked with wide eyes and open mouth].

BACKGROUND: [Clean, high-contrast gradient / cinematic environment / solid color].
[Color direction -- warm amber tones / cool blue-teal / dark moody].
Empty space on the [opposite] third for text placement.

TEXT: "[3-5 WORDS IN CAPS]" in bold white uppercase sans-serif font,
placed in the [upper-right / lower-left] area. High contrast, clearly readable
at small sizes.

STYLE: Professional YouTube thumbnail, dramatic lighting on face,
crisp and vibrant, high contrast. Sharp eyes, natural skin tones.

EXCLUDE: Watermarks, logos, text artifacts, clutter, busy backgrounds, distortion.
```

### Nano Banana for Editing Existing Frames (Most Relevant to ClipForge)

For editing real video frames rather than generating from scratch:

```python
prompt = """
Edit this video frame to look like a professional YouTube thumbnail:

1. SUBJECT: Keep the person's face, expression, and features exactly as they are.
   Enhance skin tones to be warm and vibrant but natural. Sharpen the eyes.
   Add subtle rim lighting on the edges of the person to separate from background.

2. BACKGROUND: Replace/enhance the background to be a dark, cinematic gradient
   with subtle bokeh lights. Maintain depth and atmosphere.

3. COLOR GRADE: Apply cinematic teal-and-orange color grading.
   Shadows slightly teal, highlights warm. Increase overall contrast by 20%.

4. COMPOSITION: Ensure the subject fills approximately 40-50% of the frame.
   Leave clean space for text overlay in the [left/right] third.

5. OUTPUT: 16:9 aspect ratio, 1280x720, professional and polished.
"""
```

**Important limitations:**
- Text rendering can still have line-break issues -- generate text separately via Pillow
- Complex scenes with fine details (hair, glass) may need manual refinement
- Character consistency across multiple thumbnails requires detailed re-description

### NanoThumbnail (Open Source Tool)

There's an open-source project on GitHub (`yoanbernabeu/NanoThumbnail`) that generates viral thumbnails with AI, specifically built around Nano Banana.

### A/B Testing Workflow

Generate 2 layout variations, then A/B test in YouTube Studio. This reduces decision fatigue while maintaining rigor. Production time reportedly drops by **85%** compared to manual design.

---

## 4. Comedy/Lifestyle Content -- Specific Techniques

### What Works for Comedy/Trolling/Lifestyle Thumbnails

This genre has specific visual requirements that differ from tech/business thumbnails:

### Expression Types That Convert

- **Shock/disbelief**: Wide eyes, dropped jaw -- the single most effective comedy thumbnail expression
- **Exaggerated laughter**: Big, contagious, open-mouth laugh
- **Frustration/pain**: Hands on head, squinted eyes (great for "gone wrong" content)
- **Confusion**: Squinted eyes, tilted head, one eyebrow up
- **Side-eye**: Skeptical look, works great for reaction content

**Key data point**: Even small tweaks like making eyes wider or smiles bigger make a real difference. Subtlety does not translate on YouTube -- if you're shocked, look REALLY shocked.

### Color Strategies for Comedy

- **Bright, saturated backgrounds** work better than dark/moody for comedy (opposite of tech/business)
- **Solid color or gradient backgrounds** (bright yellow, electric blue, hot pink) are more effective than realistic scenes
- **High contrast between subject and background** is non-negotiable
- Red and yellow are the most attention-grabbing colors in feeds

### Text Strategies for Comedy/Lifestyle

- Short, punchy, and specific to the video's core joke or situation
- Numbers work: "$200 GONE", "DAY 7...", "24 HOURS"
- Ellipsis creates open loops: "DAY 15...", "MEAL 12..."
- Implied conflict: "SHE LEFT", "THEY LIED", "I QUIT"
- Direct quotes work for drama: "I CAN'T BELIEVE IT"

### Layout Patterns

The dominant pattern for comedy/lifestyle thumbnails:

```
+------------------------------------------+
|                    |                      |
|   TEXT OVERLAY     |                      |
|   (3-5 words,     |    FACE CLOSE-UP     |
|    bold, caps)     |    (exaggerated      |
|                    |     expression)      |
|   [optional:       |                      |
|    emoji/arrow]    |                      |
+------------------------------------------+
```

- Face on the right third, text on the left third (or vice versa)
- Face should be a **tight close-up** -- head and shoulders minimum, preferably just head
- Background should be clean and non-distracting (blur, solid color, or gradient)
- Optional: 1-2 emojis for emphasis (fire, skull, eyes)
- Optional: Red arrow pointing at key element

### Reaction Channel Specific

- Highlight face expressions with bold directional brackets or arrows
- Limit visual cues to 1-2 per thumbnail to avoid clutter
- Feature a large, high-quality cutout of the creator's face displaying dramatic expression

---

## 5. Concrete Recommendations for ClipForge

### Priority 1: Local Frame Enhancement (Free, Immediate Impact)

Add a `enhance_frame()` function that runs before Pillow compositing:

1. **CLAHE** on luminance channel (contrast without blowout)
2. **Saturation boost** (1.2-1.4x via HSV or Pillow ImageEnhance)
3. **Unsharp mask** (sharpening, especially important for video frames which are softer than photos)
4. **Background darkening** using existing face detection mask (darken background 30-50%, leave subject untouched)
5. **Vignette** (darken edges to draw eye to center)

This alone will make thumbnails look dramatically better because video frames are inherently flat and soft compared to photographs.

### Priority 2: Subject Isolation Pipeline

Add `rembg` (with `birefnet-general` model) to the pipeline:

1. Extract subject from frame using rembg
2. Enhance subject separately (sharpen, color correct)
3. Create styled background (gradient, blur of original scene, or solid color)
4. Composite subject onto background
5. Add Pillow text overlays on top

This creates the "professional studio" look that separates good thumbnails from great ones.

### Priority 3: Upgrade Gemini Integration

**Model upgrade**: Change from `gemini-2.0-flash-exp-image-generation` to `gemini-2.5-flash-image` (production-ready) or `gemini-3-pro-image-preview` (best quality).

**Add image_config**: Use the new API parameters for aspect ratio and resolution:

```python
config=types.GenerateContentConfig(
    response_modalities=["TEXT", "IMAGE"],
    image_config=types.ImageConfig(
        aspect_ratio="16:9",
        image_size="2K",
    ),
),
```

**Improve the enhancement prompt**: The current prompt tries to do too much (full background replacement). A lighter touch works better:

```python
enhancement_prompt = f"""
Enhance this YouTube thumbnail for the video "{video_title}":

1. Make the subject's face more vibrant -- slightly warmer skin tones,
   sharper eyes, brighter catch lights. Do NOT change their identity or features.

2. Increase overall contrast by ~20%. Push shadows darker, highlights brighter.

3. Apply subtle cinematic color grading: teal-shifted shadows, warm highlights.

4. Add a subtle rim light glow on the subject's edges for background separation.

5. If there is text in the image, keep it EXACTLY as-is -- same words,
   position, colors, and size.

6. Output at exactly 1280x720 pixels.

IMPORTANT: The result should look like the same image, just more polished
and professional. Do NOT dramatically change the composition or generate
new elements.
"""
```

### Priority 4: Create a "House Style" LUT

Create a custom FFmpeg LUT that defines ClipForge's visual signature:

1. Generate identity LUT: `ffmpeg -f lavfi -i haldclutsrc=8 identity.png`
2. Edit in image editor to create the desired color grade
3. Apply to all frames before compositing for consistent, branded look

### Priority 5: Comedy-Specific Thumbnail Mode

Since the target users make comedy/lifestyle/trolling content, add a `comedy` mode that:

- Defaults to **brighter, more saturated backgrounds** instead of dark/moody
- Prefers solid color or gradient backgrounds
- Uses the `playful` or `condensed` font style by default
- Cranks up saturation and contrast more aggressively
- Sizes faces larger in the composition (50-60% of frame instead of 40%)
- Defaults to including an emoji overlay

### Summary: Recommended Enhancement Stack

```
Raw Video Frame
    |
    v
[1] OpenCV/Pillow local enhancement (CLAHE + sharpen + saturation + grade)
    |
    v
[2] rembg subject isolation (optional, when face detected)
    |
    v
[3] Background treatment (blur/darken/gradient/solid based on concept)
    |
    v
[4] Pillow compositing (text, arrows, emojis, borders -- existing code)
    |
    v
[5] Gemini AI polish (optional -- upgrade to 2.5-flash or 3-pro)
    |
    v
Final 1280x720 Thumbnail
```

Each layer is independently valuable. Even steps 1 + 4 alone (no AI cost) will produce dramatically better results than the current raw-frame + Pillow approach.

---

## Sources

- [YouTube Thumbnail Design Tips (vidIQ, 2026)](https://vidiq.com/blog/post/youtube-thumbnail-design-tips/)
- [Best Practices for YouTube Thumbnails (ThumbnailTest, 2026)](https://thumbnailtest.com/guides/best-practices-youtube-thumbnail/)
- [Thumbnail Design Principles: 2026 Conversion Guide (ThumbMagic)](https://www.thumbmagic.co/blog/thumbnail-design-principles)
- [YouTube Thumbnail Tips (NexLev, 2026)](https://www.nexlev.io/youtube-thumbnail-tips)
- [Gemini API Image Generation Docs](https://ai.google.dev/gemini-api/docs/image-generation)
- [How to Prompt Gemini 2.5 Flash Image (Google Developers Blog)](https://developers.googleblog.com/en/how-to-prompt-gemini-2-5-flash-image-generation-for-the-best-results/)
- [Gemini Image Prompt Guide (Google DeepMind)](https://deepmind.google/models/gemini-image/prompt-guide/)
- [Nano Banana Pro Prompting Guide (DEV Community / Google AI)](https://dev.to/googleai/nano-banana-pro-prompting-guide-strategies-1h9n)
- [Nano Banana YouTube Thumbnails Guide (BananaThumbnail)](https://blog.bananathumbnail.com/can-nano-bananas-make-youtube-thumbnails/)
- [10 Nano Banana Prompts for YouTube (VidPros)](https://vidpros.com/10-nano-banana-prompts-for-clickable-youtube-thumbnails/)
- [Gemini Image API Guide 2026 (LaoZhang)](https://blog.laozhang.ai/en/posts/gemini-image-api-guide-2026)
- [AI Photo Enhancer with Gemini (DEV Community)](https://dev.to/abdellahhallou/how-i-built-an-ai-photo-enhancer-that-makes-your-selfies-less-tragic-using-google-gemini-python-1c40)
- [FFmpeg Color Grading with LUTs (Gabor Heja)](https://gabor.heja.hu/blog/2024/12/10/using-ffmpeg-to-color-correct-color-grade-a-video-lut-hald-clut/)
- [OpenCV CLAHE (PyImageSearch)](https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/)
- [OpenCV Image Enhancement (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/image-enhancement-techniques-using-opencv-python/)
- [PicTex Text Effects Tutorial (DEV Community)](https://dev.to/francozanardi/python-tutorial-adding-shadows-and-outlines-to-text-on-images-1n9a)
- [rembg Background Removal (GitHub)](https://github.com/danielgatis/rembg)
- [YouTube Face: The Strategy That Works (ThumbnailMaker)](https://www.thumbnailmaker.co/post/youtube-thumbnail-face)
- [5 YouTube Thumbnail Pose Types That Get Clicks (1of10)](https://1of10.com/blog/youtube-thumbnail-pose/)
- [The Rise of YouTube Face (The Ringer, 2026)](https://www.theringer.com/2026/03/09/pop-culture/youtube-face-thumbnails-history-explained)
- [YouTube Thumbnail Reaction Face Generator (PhotoAI)](https://photoai.com/youtube-thumbnail-reaction-face)
- [AI YouTube Thumbnail Generators (Juma/Team-GPT, 2026)](https://juma.ai/blog/ai-youtube-thumbnail-generators)
- [NanoThumbnail (GitHub)](https://github.com/yoanbernabeu/NanoThumbnail)
- [Gemini 2.5 Flash Image Production Announcement (Google Developers Blog)](https://developers.googleblog.com/en/gemini-2-5-flash-image-now-ready-for-production-with-new-aspect-ratios/)
- [Gemini 3 Pro Image / Nano Banana Pro (Google DeepMind)](https://deepmind.google/models/gemini-image/pro/)
- [Introducing Gemini 2.5 Flash Image (Google Developers Blog)](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/)
