#!/usr/bin/env python3
"""
Verify that frame extraction covers the ENTIRE video, not just the beginning.
Shows timestamp distribution across the video duration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

VIDEO_URL = "https://youtu.be/5MTgirUykUM"

from backend.services.video_ingestion import download_video
from backend.services.frame_extraction import extract_candidate_frames

print("=" * 70)
print("FRAME EXTRACTION COVERAGE VERIFICATION")
print("=" * 70)

# Download video
print(f"\nDownloading video: {VIDEO_URL}")
video_info = download_video(VIDEO_URL, "verify_test")
duration = video_info['duration']
print(f"Video title: {video_info['title']}")
print(f"Total duration: {duration}s ({duration/60:.1f} minutes)")

# Extract frames
print(f"\nExtracting frames...")
frames = extract_candidate_frames(video_info["video_path"], "verify_test")

# Analyze timestamp distribution
print(f"\n{'=' * 70}")
print(f"EXTRACTED {len(frames)} FRAMES - TIMESTAMP ANALYSIS")
print(f"{'=' * 70}\n")

timestamps = [f['timestamp'] for f in frames]
timestamps.sort()

print("Frame Distribution:")
print("-" * 70)
print(f"{'Frame #':<10} {'Timestamp':<15} {'% of Video':<15} {'Source':<20}")
print("-" * 70)

for i, frame in enumerate(frames, 1):
    ts = frame['timestamp']
    percent = (ts / duration) * 100
    source = frame.get('source', 'unknown')
    print(f"{i:<10} {ts:>6.2f}s{'':<8} {percent:>5.1f}%{'':<9} {source:<20}")

print("-" * 70)

# Statistics
print(f"\nSTATISTICS:")
print(f"  First frame: {timestamps[0]:.2f}s (at {timestamps[0]/duration*100:.1f}% of video)")
print(f"  Last frame:  {timestamps[-1]:.2f}s (at {timestamps[-1]/duration*100:.1f}% of video)")
print(f"  Coverage:    {timestamps[-1] - timestamps[0]:.2f}s ({(timestamps[-1] - timestamps[0])/duration*100:.1f}% of video)")
print(f"  Average gap: {(timestamps[-1] - timestamps[0])/(len(frames)-1):.2f}s between frames")

# Check for concentration in beginning
early_frames = sum(1 for ts in timestamps if ts < duration * 0.33)  # First third
mid_frames = sum(1 for ts in timestamps if duration * 0.33 <= ts < duration * 0.66)  # Middle third
late_frames = sum(1 for ts in timestamps if ts >= duration * 0.66)  # Last third

print(f"\nDISTRIBUTION BY VIDEO SECTION:")
print(f"  First third  (0-33%):   {early_frames} frames ({early_frames/len(frames)*100:.1f}%)")
print(f"  Middle third (33-66%):  {mid_frames} frames ({mid_frames/len(frames)*100:.1f}%)")
print(f"  Last third   (66-100%): {late_frames} frames ({late_frames/len(frames)*100:.1f}%)")

# Verdict
print(f"\n{'=' * 70}")
if late_frames > 0 and timestamps[-1] > duration * 0.8:
    print("✅ VERIFIED: Frames extracted from ENTIRE video including the end!")
else:
    print("⚠️  WARNING: Frames may be concentrated in the beginning of the video")
print(f"{'=' * 70}")
