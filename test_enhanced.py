#!/usr/bin/env python3
"""Quick test to see Claude's style_notes for enhanced thumbnails"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.thumbnail_generator import analyze_frames_with_claude

# Use existing frames
frames = [
    {"file_path": f"data/frames/test_job/scene_{i:03d}.jpg", "frame_index": i}
    for i in range(10)
]

# Simple test
concepts = analyze_frames_with_claude(
    "Rick Astley - Never Gonna Give You Up - Classic 80s music video",
    frames,
    "Never Gonna Give You Up (4K Remaster)"
)

print("\n=== ENHANCED CONCEPTS ===\n")
for i, concept in enumerate(concepts, 1):
    print(f"Concept {i}:")
    print(f"  Text: {concept.get('text_overlay')}")
    print(f"  Position: {concept.get('text_position')}")
    print(f"  Text Color: {concept.get('text_color')}")
    print(f"  Stroke Color: {concept.get('text_stroke_color')}")
    print(f"  STYLE NOTES: {concept.get('style_notes')}")  # <<<< THIS IS KEY
    print(f"  CTR Tier: {concept.get('estimated_ctr_tier')}")
    print(f"  Reasoning: {concept.get('reasoning')}")
    print()
