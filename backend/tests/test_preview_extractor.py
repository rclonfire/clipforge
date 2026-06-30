"""
Tests for Task 1 (02-03): Preview extractor service.
TDD RED phase — defines expected behavior before implementation.

Tests:
    1. test_ffmpeg_command_order — -ss appears before -i in the subprocess call
    2. test_pre_post_roll_margins — pre-roll (1s) and post-roll (0.5s) applied correctly
    3. test_pre_roll_clamps_at_zero — start cannot go below 0.0
    4. test_returns_output_path — function returns the output_path string on success
"""
from unittest.mock import patch, MagicMock

import pytest


class TestExtractPreview:
    """Unit tests for extract_preview() FFmpeg wrapper."""

    def test_ffmpeg_command_order(self):
        """
        -ss must appear at index 1 (BEFORE -i) for fast input seek.
        -i must follow -ss, then -t, -c copy, -movflags +faststart.
        """
        from backend.services.preview_extractor import extract_preview

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            extract_preview(
                video_path="/tmp/test.mp4",
                start_seconds=10.0,
                end_seconds=40.0,
                output_path="/tmp/out.mp4",
                ffmpeg_path="/opt/homebrew/bin/ffmpeg",
            )

        call_args = mock_run.call_args
        cmd = call_args[0][0]  # first positional arg is the cmd list

        # -ss must appear BEFORE -i (input seek, not output seek)
        assert "-ss" in cmd, "-ss flag must be present in FFmpeg command"
        assert "-i" in cmd, "-i flag must be present in FFmpeg command"

        ss_index = cmd.index("-ss")
        i_index = cmd.index("-i")

        assert ss_index < i_index, (
            f"-ss (index {ss_index}) must come BEFORE -i (index {i_index}) for fast seek"
        )
        # -y is at index 1, so -ss is at index 2 (still BEFORE -i at index 4)
        assert ss_index == 2, f"-ss must be at index 2 (after -y), got {ss_index}"

        # -c copy for stream copy (no re-encode)
        assert "-c" in cmd, "-c flag must be present"
        c_index = cmd.index("-c")
        assert cmd[c_index + 1] == "copy", "-c must be followed by 'copy'"

        # -movflags +faststart for browser streaming
        assert "-movflags" in cmd, "-movflags flag must be present"
        mf_index = cmd.index("-movflags")
        assert cmd[mf_index + 1] == "+faststart", "-movflags must be followed by '+faststart'"

    def test_pre_post_roll_margins(self):
        """
        1-second pre-roll: adj_start = start_seconds - 1.0 = 10.0 - 1.0 = 9.0
        0.5-second post-roll: adj_end = end_seconds + 0.5 = 40.0 + 0.5 = 40.5
        duration = adj_end - adj_start = 40.5 - 9.0 = 31.5
        """
        from backend.services.preview_extractor import extract_preview

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            extract_preview(
                video_path="/tmp/test.mp4",
                start_seconds=10.0,
                end_seconds=40.0,
                output_path="/tmp/out.mp4",
                ffmpeg_path="/opt/homebrew/bin/ffmpeg",
            )

        cmd = mock_run.call_args[0][0]

        ss_index = cmd.index("-ss")
        adj_start_str = cmd[ss_index + 1]
        assert float(adj_start_str) == 9.0, (
            f"Expected -ss 9.0 (10.0 - 1.0 pre-roll), got {adj_start_str}"
        )

        t_index = cmd.index("-t")
        duration_str = cmd[t_index + 1]
        assert float(duration_str) == 31.5, (
            f"Expected -t 31.5 (40.5 - 9.0), got {duration_str}"
        )

    def test_pre_roll_clamps_at_zero(self):
        """
        When start_seconds=0.5, adj_start = max(0.0, 0.5 - 1.0) = 0.0 (clamped, not -0.5).
        """
        from backend.services.preview_extractor import extract_preview

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            extract_preview(
                video_path="/tmp/test.mp4",
                start_seconds=0.5,
                end_seconds=30.0,
                output_path="/tmp/out.mp4",
                ffmpeg_path="/opt/homebrew/bin/ffmpeg",
            )

        cmd = mock_run.call_args[0][0]

        ss_index = cmd.index("-ss")
        adj_start_str = cmd[ss_index + 1]
        assert float(adj_start_str) == 0.0, (
            f"Expected -ss 0.0 (clamped from -0.5), got {adj_start_str}"
        )

    def test_returns_output_path(self):
        """
        On subprocess success, extract_preview must return the output_path string passed in.
        """
        from backend.services.preview_extractor import extract_preview

        mock_result = MagicMock()
        mock_result.returncode = 0

        output_path = "/tmp/clip_preview.mp4"

        with patch("subprocess.run", return_value=mock_result):
            result = extract_preview(
                video_path="/tmp/test.mp4",
                start_seconds=10.0,
                end_seconds=40.0,
                output_path=output_path,
                ffmpeg_path="/opt/homebrew/bin/ffmpeg",
            )

        assert result == output_path, (
            f"Expected return value '{output_path}', got '{result}'"
        )
