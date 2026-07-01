#!/usr/bin/env python3
"""
Build or inspect the ClipForge creator-style profile.

The profile captures the patterns from the videos a creator likes/engages with,
so generated post copy reads like what already performs in their niche. It's
saved to data/style_profile.json and used automatically by the post-prep feature.

Usage:
    # Derive a profile from a JSON list of liked/engaged videos (needs ANTHROPIC_API_KEY)
    python build_style_profile.py --from liked_videos.json --niche "natural hair / faith"

    # Print the current profile
    python build_style_profile.py --show

liked_videos.json is a list of objects, e.g.:
    [
      {"platform": "youtube", "channel": "...", "title": "...",
       "caption": "...", "hashtags": ["#..."], "url": "..."},
      {"platform": "tiktok", "title": "...", "hashtags": ["#fyp", "#..."]}
    ]
"""
from __future__ import annotations

import argparse
import json
import sys

from backend.services.creator_style import (
    derive_style_profile,
    load_style_profile,
    save_style_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/inspect the creator-style profile")
    parser.add_argument("--from", dest="source", help="JSON file of liked/engaged videos")
    parser.add_argument("--niche", help="Niche hint to anchor the analysis")
    parser.add_argument("--show", action="store_true", help="Print the current saved profile")
    args = parser.parse_args()

    if args.show:
        profile = load_style_profile()
        if not profile:
            print("No style profile saved yet.")
            return 1
        print(json.dumps(profile, indent=2))
        return 0

    if not args.source:
        parser.error("provide --from <file.json> to build a profile, or --show to inspect")

    with open(args.source, encoding="utf-8") as f:
        reference_items = json.load(f)
    if not isinstance(reference_items, list) or not reference_items:
        print(f"Expected a non-empty JSON list in {args.source}", file=sys.stderr)
        return 1

    profile = derive_style_profile(reference_items, niche=args.niche)
    path = save_style_profile(profile)
    print(f"Saved style profile -> {path}")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
