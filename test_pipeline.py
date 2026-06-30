#!/usr/bin/env python3
"""
Test script for ClipForge thumbnail generation pipeline.
Tests frame extraction with quality scoring and Claude thumbnail generation.
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

# Test with a short public domain video or ask user for URL
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Short test video

def test_frame_extraction():
    """Test frame extraction with quality scoring."""
    logger.info("=== Testing Frame Extraction ===")

    from backend.services.video_ingestion import download_video
    from backend.services.frame_extraction import extract_candidate_frames

    # Download video
    logger.info(f"Downloading test video: {TEST_URL}")
    video_info = download_video(TEST_URL, "test_job")
    logger.info(f"Downloaded: {video_info['title']} ({video_info['duration']}s)")

    # Extract frames
    logger.info("Extracting candidate frames with quality scoring...")
    frames = extract_candidate_frames(video_info["video_path"], "test_job")

    logger.info(f"\n=== Extracted {len(frames)} frames ===")
    for i, frame in enumerate(frames[:10]):  # Show top 10
        logger.info(
            f"Frame {i+1}: t={frame['timestamp']:.2f}s | "
            f"Score={frame.get('combined_score', 0):.1f} | "
            f"Face={frame.get('face_score', 0):.1f} | "
            f"Sharp={frame.get('sharpness_score', 0):.1f} | "
            f"Light={frame.get('lighting_score', 0):.1f} | "
            f"Has Face={frame.get('has_face', False)}"
        )

    return video_info, frames


def test_thumbnail_generation(video_info, frames):
    """Test full thumbnail generation with Claude."""
    logger.info("\n=== Testing Thumbnail Generation with Claude ===")

    from backend.services.transcription import transcribe_video
    from backend.services.thumbnail_generator import (
        analyze_frames_with_claude,
        generate_thumbnail
    )

    # Get transcript (for Claude context)
    logger.info("Transcribing video for context...")
    try:
        transcript_data = transcribe_video(
            video_info["video_path"],
            youtube_url=TEST_URL
        )
        transcript = transcript_data["timestamped_transcript"]
        logger.info(f"Transcript length: {len(transcript)} chars")
    except Exception as e:
        logger.warning(f"Transcription failed: {e}. Using title only.")
        transcript = f"Video: {video_info['title']}"

    # Analyze frames with Claude
    logger.info("Sending frames to Claude for analysis...")
    try:
        concepts = analyze_frames_with_claude(
            transcript,
            frames,
            video_title=video_info["title"]
        )
        logger.info(f"\n=== Claude generated {len(concepts)} thumbnail concepts ===")

        for i, concept in enumerate(concepts):
            logger.info(f"\nConcept {i+1}:")
            logger.info(f"  Frame Index: {concept.get('frame_index')}")
            logger.info(f"  Text: {concept.get('text_overlay')}")
            logger.info(f"  Position: {concept.get('text_position')}")
            logger.info(f"  CTR Tier: {concept.get('estimated_ctr_tier')}")
            logger.info(f"  Reasoning: {concept.get('reasoning')}")

        # Generate actual thumbnails
        logger.info("\n=== Generating thumbnail images ===")
        generated = []
        for i, concept in enumerate(concepts):
            thumb_id = f"test_{i+1}"
            frame_idx = concept.get("frame_index", 0)

            if frame_idx >= len(frames):
                frame_idx = 0

            thumbnail_path = generate_thumbnail(
                frame_path=frames[frame_idx]["file_path"],
                concept=concept,
                job_id="test_job",
                thumbnail_id=thumb_id,
            )
            generated.append(thumbnail_path)
            logger.info(f"✓ Generated thumbnail {i+1}: {thumbnail_path}")

        logger.info(f"\n=== SUCCESS! Generated {len(generated)} thumbnails ===")
        logger.info(f"Thumbnails saved to: {Path(generated[0]).parent}")

        return generated

    except Exception as e:
        logger.error(f"Claude analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Run the full test pipeline."""
    logger.info("Starting ClipForge Pipeline Test")
    logger.info("=" * 60)

    try:
        # Test 1: Frame extraction
        video_info, frames = test_frame_extraction()

        # Test 2: Thumbnail generation
        thumbnails = test_thumbnail_generation(video_info, frames)

        if thumbnails:
            logger.info("\n" + "=" * 60)
            logger.info("✅ PIPELINE TEST COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Video: {video_info['title']}")
            logger.info(f"Frames extracted: {len(frames)}")
            logger.info(f"Thumbnails generated: {len(thumbnails)}")
            logger.info(f"\nView thumbnails at:")
            for thumb in thumbnails:
                logger.info(f"  - {thumb}")
        else:
            logger.error("❌ Thumbnail generation failed")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
