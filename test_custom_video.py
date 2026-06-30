#!/usr/bin/env python3
"""
Test ClipForge with a custom YouTube video URL
"""
import sys
import os
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# User's custom video URL
VIDEO_URL = "https://youtu.be/5MTgirUykUM"

def main():
    """Run the full test pipeline with custom video."""
    logger.info("=" * 70)
    logger.info("ClipForge - Custom Video Test")
    logger.info("=" * 70)
    logger.info(f"Video URL: {VIDEO_URL}\n")

    from backend.services.video_ingestion import download_video
    from backend.services.frame_extraction import extract_candidate_frames
    from backend.services.transcription import transcribe_video
    from backend.services.thumbnail_generator import (
        analyze_frames_with_claude,
        generate_thumbnail
    )

    try:
        # Step 1: Download video
        logger.info("Step 1/4: Downloading video...")
        video_info = download_video(VIDEO_URL, "custom_test")
        logger.info(f"✓ Downloaded: {video_info['title']}")
        logger.info(f"  Duration: {video_info['duration']}s\n")

        # Step 2: Extract frames with quality scoring
        logger.info("Step 2/4: Extracting candidate frames with AI scoring...")
        frames = extract_candidate_frames(video_info["video_path"], "custom_test")
        logger.info(f"✓ Extracted {len(frames)} frames")
        logger.info(f"  Top frame score: {frames[0].get('combined_score', 0):.1f}/100")
        logger.info(f"  Frames with faces: {sum(1 for f in frames if f.get('has_face'))}\n")

        # Step 3: Get transcript
        logger.info("Step 3/4: Transcribing video...")
        try:
            transcript_data = transcribe_video(
                video_info["video_path"],
                youtube_url=VIDEO_URL
            )
            transcript = transcript_data["timestamped_transcript"]
            logger.info(f"✓ Transcript: {len(transcript)} characters\n")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}. Using title only.")
            transcript = f"Video: {video_info['title']}"

        # Step 4: Generate thumbnails with Claude
        logger.info("Step 4/4: Generating thumbnails with Claude AI...")
        concepts = analyze_frames_with_claude(
            transcript,
            frames,
            video_title=video_info["title"]
        )

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Claude Generated {len(concepts)} Thumbnail Concepts:")
        logger.info(f"{'=' * 70}\n")

        thumbnails = []
        for i, concept in enumerate(concepts, 1):
            logger.info(f"Concept {i}:")
            logger.info(f"  Text: {concept.get('text_overlay') or 'NO TEXT (face-only)'}")
            logger.info(f"  Position: {concept.get('text_position', 'N/A')}")
            logger.info(f"  Colors: {concept.get('text_color', 'N/A')} on {concept.get('text_stroke_color', 'N/A')}")
            logger.info(f"  Style: {concept.get('style_notes')}")
            logger.info(f"  CTR Tier: {concept.get('estimated_ctr_tier').upper()}")
            logger.info(f"  Why: {concept.get('reasoning')}")

            # Generate thumbnail
            frame_idx = concept.get("frame_index", 0)
            if frame_idx >= len(frames):
                frame_idx = 0

            thumb_id = f"concept_{i}"
            thumbnail_path = generate_thumbnail(
                frame_path=frames[frame_idx]["file_path"],
                concept=concept,
                job_id="custom_test",
                thumbnail_id=thumb_id,
            )
            thumbnails.append(thumbnail_path)
            logger.info(f"  ✓ Generated: {thumb_id}.jpg\n")

        # Summary
        logger.info(f"{'=' * 70}")
        logger.info("✅ COMPLETE! Generated 2026-Optimized Thumbnails")
        logger.info(f"{'=' * 70}")
        logger.info(f"Video: {video_info['title']}")
        logger.info(f"Frames analyzed: {len(frames)}")
        logger.info(f"Thumbnails created: {len(thumbnails)}")
        logger.info(f"\nThumbnails saved to:")
        logger.info(f"  {Path(thumbnails[0]).parent}/")
        logger.info(f"\nOpening folder...")

        # Open folder
        import subprocess
        subprocess.run(["open", str(Path(thumbnails[0]).parent)])

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
