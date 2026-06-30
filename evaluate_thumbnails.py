#!/usr/bin/env python3
"""
Comprehensive thumbnail evaluation tool.
Analyzes thumbnails like a real YouTube viewer would see them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2

THUMBS_DIR = Path("data/thumbnails/custom_test")

def analyze_thumbnail(image_path):
    """Comprehensive analysis of a thumbnail."""
    img = Image.open(image_path)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1. File size and quality metrics
    file_size = Path(image_path).stat().st_size / 1024  # KB
    width, height = img.size

    # 2. Color vibrancy (saturation analysis)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    vibrancy_score = min(100, (saturation / 128) * 100)

    # 3. Contrast analysis
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    contrast_score = min(100, (contrast / 60) * 100)

    # 4. Sharpness analysis
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(100, (laplacian_var / 500) * 100)

    # 5. Edge density (visual complexity)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)
    complexity_score = min(100, (edge_density / 0.15) * 100)

    # 6. Brightness analysis
    brightness = np.mean(gray)
    brightness_score = 100 * (1 - abs(brightness - 140) / 140)
    brightness_score = max(0, min(100, brightness_score))

    return {
        "file_size_kb": file_size,
        "dimensions": f"{width}x{height}",
        "vibrancy": vibrancy_score,
        "contrast": contrast_score,
        "sharpness": sharpness_score,
        "complexity": complexity_score,
        "brightness": brightness_score,
        "overall": (vibrancy_score + contrast_score + sharpness_score) / 3
    }


def create_mobile_preview(image_path, output_path):
    """Create a mobile-sized preview (320x180) to see actual mobile appearance."""
    img = Image.open(image_path)

    # Resize to mobile thumbnail size
    mobile = img.resize((320, 180), Image.LANCZOS)

    # Create comparison image: original vs mobile
    comparison = Image.new('RGB', (1280 + 40 + 320, max(720, 180)), 'white')

    # Paste original on left
    comparison.paste(img, (0, 0))

    # Paste mobile on right (centered vertically)
    mobile_y = (720 - 180) // 2
    comparison.paste(mobile, (1280 + 40, mobile_y))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 730), "Desktop (1280x720)", fill='black', font=font)
    draw.text((1280 + 50, 730), "Mobile (320x180)", fill='black', font=font)
    draw.text((1280 + 50, mobile_y - 30), "ACTUAL SIZE →", fill='red', font=font)

    comparison.save(output_path, "JPEG", quality=95)
    return str(output_path)


def create_evaluation_grid(thumbs_dir):
    """Create a side-by-side comparison grid with scores."""
    thumbs = sorted(Path(thumbs_dir).glob("concept_*.jpg"))

    if len(thumbs) == 0:
        print("No thumbnails found!")
        return None

    print("=" * 90)
    print("THUMBNAIL EVALUATION REPORT".center(90))
    print("=" * 90)
    print()

    print(f"Analyzing {len(thumbs)} thumbnails...\n")

    # Analyze each thumbnail
    results = []
    for i, thumb_path in enumerate(thumbs, 1):
        analysis = analyze_thumbnail(thumb_path)
        results.append((thumb_path.name, analysis))

        print(f"{'─' * 90}")
        print(f"THUMBNAIL {i}: {thumb_path.name}")
        print(f"{'─' * 90}")
        print(f"  File Size:    {analysis['file_size_kb']:.0f} KB")
        print(f"  Dimensions:   {analysis['dimensions']}")
        print()
        print(f"  QUALITY SCORES:")
        print(f"    Vibrancy:   {'█' * int(analysis['vibrancy']/5)} {analysis['vibrancy']:.1f}/100")
        print(f"    Contrast:   {'█' * int(analysis['contrast']/5)} {analysis['contrast']:.1f}/100")
        print(f"    Sharpness:  {'█' * int(analysis['sharpness']/5)} {analysis['sharpness']:.1f}/100")
        print(f"    Complexity: {'█' * int(analysis['complexity']/5)} {analysis['complexity']:.1f}/100")
        print(f"    Brightness: {'█' * int(analysis['brightness']/5)} {analysis['brightness']:.1f}/100")
        print()
        print(f"  OVERALL SCORE: {'█' * int(analysis['overall']/5)} {analysis['overall']:.1f}/100")
        print()

    # Summary
    print("=" * 90)
    print("SUMMARY".center(90))
    print("=" * 90)

    avg_overall = np.mean([r[1]['overall'] for r in results])
    avg_vibrancy = np.mean([r[1]['vibrancy'] for r in results])
    avg_contrast = np.mean([r[1]['contrast'] for r in results])
    avg_sharpness = np.mean([r[1]['sharpness'] for r in results])
    avg_size = np.mean([r[1]['file_size_kb'] for r in results])

    print()
    print(f"  Average Overall Score: {avg_overall:.1f}/100")
    print(f"  Average Vibrancy:      {avg_vibrancy:.1f}/100")
    print(f"  Average Contrast:      {avg_contrast:.1f}/100")
    print(f"  Average Sharpness:     {avg_sharpness:.1f}/100")
    print(f"  Average File Size:     {avg_size:.0f} KB")
    print()

    # Recommendations
    print("=" * 90)
    print("RECOMMENDATIONS".center(90))
    print("=" * 90)
    print()

    if avg_vibrancy < 50:
        print("  ⚠️  LOW VIBRANCY - Increase color saturation for more eye-catching thumbnails")
    elif avg_vibrancy > 80:
        print("  ✅ EXCELLENT VIBRANCY - Colors are vibrant and eye-catching!")

    if avg_contrast < 50:
        print("  ⚠️  LOW CONTRAST - Increase contrast for better visibility")
    elif avg_contrast > 70:
        print("  ✅ EXCELLENT CONTRAST - Text and elements pop nicely!")

    if avg_sharpness < 40:
        print("  ⚠️  LOW SHARPNESS - Images appear soft, increase sharpening")
    elif avg_sharpness > 60:
        print("  ✅ EXCELLENT SHARPNESS - Images are crisp and clear!")

    if avg_size < 500:
        print("  ⚠️  SMALL FILE SIZE - May lack quality, consider higher quality settings")
    elif avg_size > 1000:
        print("  ✅ MAXIMUM QUALITY - Files are high-quality professional grade!")

    print()
    print("=" * 90)

    # Best thumbnail
    best = max(results, key=lambda x: x[1]['overall'])
    print(f"\n🏆 BEST THUMBNAIL: {best[0]} (Score: {best[1]['overall']:.1f}/100)")
    print()

    return results


def create_mobile_previews_for_all(thumbs_dir):
    """Create mobile previews for all thumbnails."""
    thumbs = sorted(Path(thumbs_dir).glob("concept_*.jpg"))
    output_dir = Path(thumbs_dir) / "mobile_previews"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 90)
    print("CREATING MOBILE PREVIEWS (320x180px - ACTUAL VIEWING SIZE)".center(90))
    print("=" * 90)
    print()

    for thumb in thumbs:
        output = output_dir / f"preview_{thumb.name}"
        create_mobile_preview(thumb, output)
        print(f"  ✓ Created: {output.name}")

    print()
    print(f"Mobile previews saved to: {output_dir}")
    print()

    return output_dir


def main():
    if not THUMBS_DIR.exists():
        print(f"Error: {THUMBS_DIR} not found!")
        return

    # Run evaluation
    results = create_evaluation_grid(THUMBS_DIR)

    # Create mobile previews
    preview_dir = create_mobile_previews_for_all(THUMBS_DIR)

    print("=" * 90)
    print()
    print("EVALUATION COMPLETE!")
    print()
    print(f"📊 View mobile previews: {preview_dir}")
    print()
    print("Opening mobile previews folder...")

    import subprocess
    subprocess.run(["open", str(preview_dir)])


if __name__ == "__main__":
    main()
