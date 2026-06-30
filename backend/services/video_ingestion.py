"""
Video ingestion service for ClipForge.

Wraps yt-dlp with a structured error taxonomy so callers receive actionable
error codes rather than raw yt-dlp stderr text.

Error taxonomy:
    bot_detected       — YouTube requested sign-in / rate limit / SABR
    video_unavailable  — Video deleted or 404
    private_video      — Video is private or members-only
    geo_restricted     — Not available in the caller's country
    network_error      — Connection timeout or any other unclassified error

Security:
    URL validation rejects non-YouTube domains BEFORE yt-dlp is called,
    preventing Server-Side Request Forgery (SSRF) attacks.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from backend.config import DOWNLOADS_DIR, MAX_VIDEO_DURATION, FFPROBE_PATH

# Resolve yt-dlp binary: try venv first, then PATH lookup
_VENV_BIN = Path(sys.executable).parent
_VENV_YT_DLP = _VENV_BIN / "yt-dlp"
YT_DLP = str(_VENV_YT_DLP) if _VENV_YT_DLP.exists() else (shutil.which("yt-dlp") or "yt-dlp")

logger = logging.getLogger(__name__)

# Allowed YouTube domains for SSRF prevention
_YOUTUBE_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


class DownloadError(Exception):
    """Raised when yt-dlp fails to download a video.

    Attributes:
        error_code: One of ("bot_detected", "video_unavailable", "private_video",
                            "geo_restricted", "network_error").
        message: Human-readable description of the failure.
    """

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover
        return f"DownloadError(error_code={self.error_code!r}, message={self.message!r})"


def _classify_stderr(stderr: str) -> str:
    """Map yt-dlp stderr text to a structured error code.

    Priority order matters — check bot signals before video_unavailable
    because some bot-detection messages also contain "unavailable".
    """
    # bot_detected — must be checked before video_unavailable
    if any(pat in stderr for pat in ("Sign in to confirm", "429", "SABR")):
        return "bot_detected"

    # video_unavailable
    if any(pat in stderr for pat in ("Video unavailable", "HTTP Error 404")):
        return "video_unavailable"

    # private_video
    if any(pat in stderr for pat in ("Private video", "members-only")):
        return "private_video"

    # geo_restricted
    if "not available in your country" in stderr:
        return "geo_restricted"

    # Everything else
    return "network_error"


def _validate_youtube_url(url: str) -> None:
    """Raise DownloadError if the URL is not a recognised YouTube domain.

    This prevents SSRF by ensuring yt-dlp is only ever invoked with
    legitimate YouTube URLs.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc not in _YOUTUBE_DOMAINS:
            raise DownloadError(
                f"URL is not a recognised YouTube domain: {url!r} (netloc={netloc!r}). "
                "Only youtube.com and youtu.be are accepted.",
                error_code="video_unavailable",
            )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(
            f"Invalid URL: {url!r} — {exc}",
            error_code="network_error",
        ) from exc


def download_video(youtube_url: str, job_id: str) -> dict:
    """Download a YouTube video using yt-dlp.

    Args:
        youtube_url: A youtube.com or youtu.be URL.
        job_id: Unique job identifier used to create the output directory.

    Returns:
        dict with keys:
            video_path (str): Absolute path to the downloaded .mp4 file.
            title (str): Video title from YouTube metadata.
            duration_seconds (int): Video duration in seconds.
            thumbnail_url (str | None): YouTube thumbnail URL, or None.

    Raises:
        DownloadError: With .error_code set to one of the taxonomy values.
    """
    # SSRF prevention: validate URL domain BEFORE calling yt-dlp
    _validate_youtube_url(youtube_url)

    output_dir = DOWNLOADS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "video.%(ext)s")

    # yt-dlp download command
    # - bestvideo[height<=720]+bestaudio: 720p max — saves bandwidth; good enough for thumbnails
    # - --no-playlist: never download entire playlists from a single video URL
    # - --sleep-interval 1: reduces rate-limiting risk
    download_cmd = [
        YT_DLP,
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--sleep-interval", "1",
        "--write-info-json",
        youtube_url,
    ]

    logger.info(f"[{job_id}] Starting download: {youtube_url}")

    try:
        result = subprocess.run(
            download_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(
            f"Download timed out for job {job_id}: {youtube_url}",
            error_code="network_error",
        ) from exc

    stderr = result.stderr or ""

    if stderr:
        logger.debug(f"[{job_id}] yt-dlp stderr: {stderr[:500]}")

    # Classify errors from stderr BEFORE checking returncode —
    # yt-dlp sometimes writes errors to stderr even on rc=0.
    if result.returncode != 0:
        error_code = _classify_stderr(stderr)
        raise DownloadError(
            f"yt-dlp exited {result.returncode} for job {job_id}: {stderr[:300]}",
            error_code=error_code,
        )

    # Find the downloaded video file
    video_files = list(output_dir.glob("video.mp4"))
    if not video_files:
        # Try any video.* extension (yt-dlp sometimes writes .mkv or .webm)
        video_files = list(output_dir.glob("video.*"))
        # Exclude .info.json files
        video_files = [f for f in video_files if f.suffix not in (".json", ".part")]

    if not video_files:
        raise DownloadError(
            f"yt-dlp succeeded (rc=0) but no video file found in {output_dir}",
            error_code="network_error",
        )

    video_path = video_files[0]

    # Read metadata from the .info.json file written by --write-info-json
    title = ""
    duration_seconds = 0
    thumbnail_url: str | None = None

    info_files = list(output_dir.glob("*.info.json"))
    if info_files:
        try:
            info = json.loads(info_files[0].read_text(encoding="utf-8"))
            title = info.get("title", "")
            duration_seconds = int(info.get("duration") or 0)
            thumbnail_url = info.get("thumbnail") or None
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[{job_id}] Failed to parse info.json: {exc}")

    logger.info(
        f"[{job_id}] Download complete: {title!r} ({duration_seconds}s) -> {video_path}"
    )

    return {
        "video_path": str(video_path),
        "title": title,
        "duration_seconds": duration_seconds,
        "thumbnail_url": thumbnail_url,
    }


def get_video_metadata(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)
