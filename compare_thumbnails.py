#!/usr/bin/env python3
"""
Create A/B comparison grid - see all thumbnails side-by-side
Like a real YouTube feed to pick the best one
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont

THUMBS_DIR = Path("data/thumbnails/custom_test")

def create_comparison_grid(thumbs_dir, output_path="data/thumbnails/comparison_grid.jpg"):
    """Create a side-by-side grid of all thumbnails."""
    thumbs = sorted(Path(thumbs_dir).glob("concept_*.jpg"))

    if len(thumbs) == 0:
        print("No thumbnails found!")
        return None

    print(f"\nCreating comparison grid for {len(thumbs)} thumbnails...")

    # Load all thumbnails
    images = []
    for thumb in thumbs:
        img = Image.open(thumb)
        # Resize to standard size for grid
        img = img.resize((640, 360), Image.LANCZOS)
        images.append((thumb.stem, img))

    # Create grid (2 columns)
    cols = 2
    rows = (len(images) + 1) // 2
    margin = 20
    label_height = 40

    grid_w = cols * 640 + (cols + 1) * margin
    grid_h = rows * (360 + label_height) + (rows + 1) * margin

    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Add each thumbnail to grid
    for idx, (name, img) in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * (640 + margin) + margin
        y = row * (360 + label_height + margin) + margin

        # Paste thumbnail
        grid.paste(img, (x, y))

        # Add label
        label_y = y + 360 + 5
        draw.text((x, label_y), f"{idx + 1}. {name.replace('concept_', 'Concept ')}",
                  fill='black', font=font)

        # Add border
        draw.rectangle([x-2, y-2, x+642, y+362], outline='gray', width=2)

    grid.save(output_path, "JPEG", quality=95)
    print(f"✓ Comparison grid saved: {output_path}")

    return output_path


def create_youtube_feed_mockup(thumbs_dir, output_path="data/thumbnails/youtube_feed.jpg"):
    """Create a mockup that looks like a real YouTube feed."""
    thumbs = sorted(Path(thumbs_dir).glob("concept_*.jpg"))[:4]  # Top 4

    if len(thumbs) == 0:
        print("No thumbnails found!")
        return None

    print(f"\nCreating YouTube feed mockup...")

    # YouTube feed dimensions
    thumb_w, thumb_h = 360, 202  # YouTube standard ratio
    spacing = 30
    title_height = 80

    feed_w = thumb_w * 2 + spacing * 3
    feed_h = thumb_h * 2 + title_height * 2 + spacing * 4 + 60

    feed = Image.new('RGB', (feed_w, feed_h), '#181818')  # YouTube dark mode bg
    draw = ImageDraw.Draw(feed)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        title_font = ImageFont.load_default()
        header_font = title_font

    # Header
    draw.text((spacing, 15), "YouTube Feed Preview (Dark Mode)", fill='white', font=header_font)

    y_offset = 60

    # Add thumbnails in 2x2 grid
    for idx, thumb in enumerate(thumbs):
        row = idx // 2
        col = idx % 2

        x = col * (thumb_w + spacing) + spacing
        y = row * (thumb_h + title_height + spacing) + y_offset

        # Load and resize thumbnail
        img = Image.open(thumb)
        img = img.resize((thumb_w, thumb_h), Image.LANCZOS)

        # Paste thumbnail
        feed.paste(img, (x, y))

        # Add fake video title
        title_y = y + thumb_h + 8
        draw.text((x, title_y), "I Became A Pro Basketball Player...",
                  fill='white', font=title_font)
        draw.text((x, title_y + 20), "1.2M views • 2 days ago",
                  fill='#aaaaaa', font=title_font)

    feed.save(output_path, "JPEG", quality=95)
    print(f"✓ YouTube feed mockup saved: {output_path}")

    return output_path


def main():
    if not THUMBS_DIR.exists():
        print(f"Error: {THUMBS_DIR} not found!")
        return

    print("=" * 70)
    print("THUMBNAIL COMPARISON TOOL")
    print("=" * 70)

    # Create comparison grid
    grid_path = create_comparison_grid(THUMBS_DIR)

    # Create YouTube feed mockup
    feed_path = create_youtube_feed_mockup(THUMBS_DIR)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Comparison Grid: {grid_path}")
    print(f"📱 YouTube Feed Mockup: {feed_path}")
    print("\nOpening comparison files...")

    import subprocess
    if grid_path:
        subprocess.run(["open", grid_path])
    if feed_path:
        subprocess.run(["open", feed_path])


if __name__ == "__main__":
    main()
