"""
Centralized configuration via pydantic-settings BaseSettings.
Loads from .env file in project root. All directory paths are absolute
and are created on first access.
"""
from __future__ import annotations

from pathlib import Path
from functools import cached_property

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the project root (two levels up from this file: backend/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Feature flags
    use_gemini_enhancement: bool = False  # When True, Pillow output is passed to Gemini for background stylization

    # Data directory — defaults to <project_root>/data
    data_dir: Path = _PROJECT_ROOT / "data"

    # Database file path (relative to data_dir by default)
    database_url: str = ""  # computed below if empty

    # Redis
    redis_url: str = "redis://localhost:6379"

    # CORS
    cors_origins: str = "http://localhost:5173"

    # Video processing constants
    max_video_duration: int = 3600  # 1 hour
    thumbnail_width: int = 1280
    thumbnail_height: int = 720
    max_candidate_frames: int = 20

    # Whisper model size
    whisper_model: str = "medium"

    # FFmpeg binary paths (Homebrew macOS defaults)
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    ffprobe_path: str = "/opt/homebrew/bin/ffprobe"

    # ---- Computed path constants ----

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def downloads_dir(self) -> Path:
        p = (self.data_dir / "downloads").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def frames_dir(self) -> Path:
        p = (self.data_dir / "frames").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def thumbnails_dir(self) -> Path:
        p = (self.data_dir / "thumbnails").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def clips_dir(self) -> Path:
        p = (self.data_dir / "clips").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def exports_dir(self) -> Path:
        p = (self.data_dir / "exports").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def db_path(self) -> Path:
        p = (self.data_dir / "db.sqlite").resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---- Derived values ----

    @property
    def effective_database_url(self) -> str:
        """Returns DATABASE_URL if set, otherwise builds from db_path."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def gemini_headshots_dir(self) -> Path:
        return _PROJECT_ROOT / ".claude" / "skills" / "youtube-thumbnail" / "assets" / "headshots"

    @property
    def fonts_dir(self) -> Path:
        return _PROJECT_ROOT / "fonts"


# Singleton instance — import this everywhere
settings = Settings()

# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# These allow existing code that imports e.g. `from backend.config import REDIS_URL`
# to keep working without a rewrite. New code should use `settings.*` directly.
# ---------------------------------------------------------------------------
BASE_DIR = _PROJECT_ROOT
DATA_DIR = settings.data_dir
DOWNLOADS_DIR = settings.downloads_dir
FRAMES_DIR = settings.frames_dir
THUMBNAILS_DIR = settings.thumbnails_dir
CLIPS_DIR = settings.clips_dir
EXPORTS_DIR = settings.exports_dir
DATABASE_URL = settings.effective_database_url
ANTHROPIC_API_KEY = settings.anthropic_api_key
GEMINI_API_KEY = settings.gemini_api_key
REDIS_URL = settings.redis_url
CORS_ORIGINS = settings.cors_origins_list
MAX_VIDEO_DURATION = settings.max_video_duration
THUMBNAIL_WIDTH = settings.thumbnail_width
THUMBNAIL_HEIGHT = settings.thumbnail_height
MAX_CANDIDATE_FRAMES = settings.max_candidate_frames
WHISPER_MODEL = settings.whisper_model
FONTS_DIR = settings.fonts_dir
FFMPEG_PATH = settings.ffmpeg_path
FFPROBE_PATH = settings.ffprobe_path
GEMINI_HEADSHOTS_DIR = settings.gemini_headshots_dir
USE_GEMINI_ENHANCEMENT = settings.use_gemini_enhancement
