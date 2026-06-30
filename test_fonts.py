#!/usr/bin/env python3
"""Test the new fonts"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.thumbnail_generator import _get_font, FONT_PATHS

print("=" * 70)
print("FONT TEST")
print("=" * 70)

print("\nConfigured font paths:")
for style, paths in FONT_PATHS.items():
    print(f"\n{style.upper()}:")
    for path in paths:
        exists = Path(path).exists() if isinstance(path, (str, Path)) else False
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")

print("\n" + "=" * 70)
print("Testing font loading:")
print("=" * 70)

for style in ["ultra", "bold", "impact"]:
    try:
        font = _get_font(style, 100)
        print(f"\n{style.upper()}: SUCCESS - Loaded {font.getname() if hasattr(font, 'getname') else 'font'}")
    except Exception as e:
        print(f"\n{style.upper()}: ERROR - {e}")
