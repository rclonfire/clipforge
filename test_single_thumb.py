#!/usr/bin/env python3
"""Test a single thumbnail to debug style issues"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

# Test concept with specific styles
concept = {
    "text_overlay": "TEST BOX",
    "text_position": "bottom-left",
    "text_color": "#FFD700",
    "text_stroke_color": "#000000",
    "style_notes": "mrbeast, text box, boost saturation"
}

from backend.services.thumbnail_generator import generate_thumbnail

# Use an existing frame
frame_path = "data/frames/custom_test/seg_0000_00.jpg"

print("=" * 70)
print("THUMBNAIL STYLE DEBUG TEST")
print("=" * 70)
print(f"\nConcept:")
print(f"  Text: {concept['text_overlay']}")
print(f"  Style Notes: {concept['style_notes']}")
print(f"  Expected: MrBeast style (ultra-bright) + gold text in colored box\n")

# Generate
output = generate_thumbnail(frame_path, concept, "debug_test", "test_1")
print(f"Generated: {output}")
print("\nOpening for inspection...")

import subprocess
subprocess.run(["open", output])
