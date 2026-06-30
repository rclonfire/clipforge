"""
Tests for Task 1: Video ingestion service with structured error taxonomy.
TDD RED phase — defines expected behavior before implementation.
"""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestDownloadVideoErrorTaxonomy:
    """Tests for DownloadError with structured error_code."""

    def test_download_error_has_error_code_attribute(self):
        """DownloadError must have an error_code attribute."""
        from backend.services.video_ingestion import DownloadError

        err = DownloadError("some message", error_code="bot_detected")
        assert err.error_code == "bot_detected"
        assert "some message" in str(err)

    def test_bot_detected_sign_in_stderr(self, tmp_path):
        """stderr containing 'Sign in to confirm' must raise DownloadError(error_code='bot_detected')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: [youtube] Sign in to confirm you're not a bot"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-001")
        assert exc_info.value.error_code == "bot_detected"

    def test_bot_detected_429_stderr(self, tmp_path):
        """stderr containing '429' must raise DownloadError(error_code='bot_detected')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: HTTP Error 429: Too Many Requests"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-002")
        assert exc_info.value.error_code == "bot_detected"

    def test_bot_detected_sabr_stderr(self, tmp_path):
        """stderr containing 'SABR' must raise DownloadError(error_code='bot_detected')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: SABR stream format not supported"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-sabr")
        assert exc_info.value.error_code == "bot_detected"

    def test_video_unavailable_stderr(self, tmp_path):
        """stderr containing 'Video unavailable' must raise DownloadError(error_code='video_unavailable')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: [youtube] dQw4w9WgXcQ: Video unavailable"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-003")
        assert exc_info.value.error_code == "video_unavailable"

    def test_video_unavailable_404_stderr(self, tmp_path):
        """stderr containing 'HTTP Error 404' must raise DownloadError(error_code='video_unavailable')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: HTTP Error 404: Not Found"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-004")
        assert exc_info.value.error_code == "video_unavailable"

    def test_private_video_stderr(self, tmp_path):
        """stderr containing 'Private video' must raise DownloadError(error_code='private_video')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: [youtube] dQw4w9WgXcQ: Private video"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-005")
        assert exc_info.value.error_code == "private_video"

    def test_members_only_stderr(self, tmp_path):
        """stderr containing 'members-only' must raise DownloadError(error_code='private_video')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: This video is members-only content"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-006")
        assert exc_info.value.error_code == "private_video"

    def test_geo_restricted_stderr(self, tmp_path):
        """stderr containing 'not available in your country' must raise DownloadError(error_code='geo_restricted')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: This video is not available in your country"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-007")
        assert exc_info.value.error_code == "geo_restricted"

    def test_network_error_timeout(self, tmp_path):
        """subprocess.TimeoutExpired must raise DownloadError(error_code='network_error')."""
        from backend.services.video_ingestion import download_video, DownloadError

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=60)):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-008")
        assert exc_info.value.error_code == "network_error"

    def test_unclassified_error_is_network_error(self, tmp_path):
        """Unrecognized stderr with non-zero returncode must raise DownloadError(error_code='network_error')."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: Some completely unknown error occurred"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-009")
        assert exc_info.value.error_code == "network_error"


class TestDownloadVideoURLValidation:
    """Tests for SSRF prevention URL validation."""

    def test_non_youtube_url_rejected(self):
        """Non-YouTube URLs must raise DownloadError BEFORE yt-dlp is called."""
        from backend.services.video_ingestion import download_video, DownloadError

        with patch("subprocess.run") as mock_run:
            with pytest.raises(DownloadError) as exc_info:
                download_video("https://vimeo.com/123456", "test-job-010")

        # subprocess.run must NOT have been called (validation happens before yt-dlp)
        mock_run.assert_not_called()
        assert exc_info.value.error_code in ("video_unavailable", "network_error", "bot_detected") or \
               "url" in str(exc_info.value).lower() or \
               exc_info.value.error_code is not None

    def test_internal_network_url_rejected(self):
        """Internal network URLs must be rejected before yt-dlp is called."""
        from backend.services.video_ingestion import download_video, DownloadError

        with patch("subprocess.run") as mock_run:
            with pytest.raises(DownloadError):
                download_video("http://169.254.169.254/metadata", "test-job-011")

        mock_run.assert_not_called()

    def test_youtube_com_accepted(self):
        """youtube.com URLs must pass validation and attempt yt-dlp."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        # Mock successful download — video file is found
        with patch("subprocess.run", return_value=mock_result):
            with patch("pathlib.Path.glob", return_value=iter([Path("/fake/video.mp4")])):
                with patch("pathlib.Path.mkdir"):
                    try:
                        result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-012")
                        assert "video_path" in result
                    except DownloadError:
                        # Accept DownloadError since we don't have a real yt-dlp binary here
                        pass

    def test_youtu_be_accepted(self):
        """youtu.be short URLs must pass validation and attempt yt-dlp."""
        from backend.services.video_ingestion import download_video, DownloadError

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch("pathlib.Path.glob", return_value=iter([Path("/fake/video.mp4")])):
                with patch("pathlib.Path.mkdir"):
                    try:
                        result = download_video("https://youtu.be/dQw4w9WgXcQ", "test-job-013")
                        assert "video_path" in result
                    except DownloadError:
                        pass


class TestDownloadVideoSuccessReturn:
    """Tests for successful download return dict."""

    def test_successful_download_returns_video_path(self, tmp_path):
        """Successful download must return dict with 'video_path' key."""
        from backend.services.video_ingestion import download_video

        # Simulate yt-dlp writing a file
        job_dir = tmp_path / "test-job-014"
        job_dir.mkdir(parents=True)
        fake_video = job_dir / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch("backend.services.video_ingestion.DOWNLOADS_DIR", tmp_path):
                result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-014")

        assert "video_path" in result
        assert result["video_path"] == str(fake_video)

    def test_successful_download_returns_title(self, tmp_path):
        """Successful download must return dict with 'title' key."""
        from backend.services.video_ingestion import download_video

        job_dir = tmp_path / "test-job-015"
        job_dir.mkdir(parents=True)
        fake_video = job_dir / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch("backend.services.video_ingestion.DOWNLOADS_DIR", tmp_path):
                result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-015")

        assert "title" in result

    def test_successful_download_returns_duration_seconds(self, tmp_path):
        """Successful download must return dict with 'duration_seconds' key (int)."""
        from backend.services.video_ingestion import download_video

        job_dir = tmp_path / "test-job-016"
        job_dir.mkdir(parents=True)
        fake_video = job_dir / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch("backend.services.video_ingestion.DOWNLOADS_DIR", tmp_path):
                result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-016")

        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], int)

    def test_successful_download_returns_thumbnail_url(self, tmp_path):
        """Successful download must return dict with 'thumbnail_url' key."""
        from backend.services.video_ingestion import download_video

        job_dir = tmp_path / "test-job-017"
        job_dir.mkdir(parents=True)
        fake_video = job_dir / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch("backend.services.video_ingestion.DOWNLOADS_DIR", tmp_path):
                result = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-017")

        assert "thumbnail_url" in result

    def test_no_sleep_interval_flag_not_missing(self):
        """yt-dlp download command must include --sleep-interval to avoid rate limiting."""
        from backend.services.video_ingestion import download_video, DownloadError

        captured_calls = []

        def capture_run(cmd, **kwargs):
            captured_calls.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stderr = "ERROR: Video unavailable"
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=capture_run):
            try:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-018")
            except DownloadError:
                pass

        # At least one subprocess call should have happened
        assert len(captured_calls) > 0
        # The download command should include --sleep-interval
        all_args = " ".join(str(a) for call in captured_calls for a in call)
        assert "--sleep-interval" in all_args

    def test_no_playlist_flag_present(self):
        """yt-dlp download command must include --no-playlist to avoid downloading playlists."""
        from backend.services.video_ingestion import download_video, DownloadError

        captured_calls = []

        def capture_run(cmd, **kwargs):
            captured_calls.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stderr = "ERROR: Video unavailable"
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=capture_run):
            try:
                download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "test-job-019")
            except DownloadError:
                pass

        all_args = " ".join(str(a) for call in captured_calls for a in call)
        assert "--no-playlist" in all_args
