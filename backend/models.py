"""
SQLAlchemy 2.x models for ClipForge.

All primary keys are UUID strings. Status values are plain strings (not enums)
to simplify migration and serialization. updated_at is required on Job for
stale job recovery in the worker startup scan.
"""
from __future__ import annotations

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.sql import func

from backend.config import settings

Base = declarative_base()


class Job(Base):
    """
    Represents one video processing job.

    Status lifecycle:
        pending -> downloading -> transcribing -> analyzing
        -> generating_thumbnails -> detecting_clips -> complete
        (any stage can transition to: failed)
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    youtube_url = Column(Text, nullable=False)
    video_title = Column(Text)
    video_duration_seconds = Column(Integer)

    # String column (not enum) — values defined in status lifecycle above
    status = Column(String, default="pending", nullable=False)

    progress_message = Column(Text, default="")
    error_message = Column(Text)

    # Autonomous song identification (for caption naming on music/violin covers)
    song_title = Column(Text)
    song_artist = Column(Text)
    song_confidence = Column(String)  # high | medium | low | none

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime)

    thumbnails = relationship("Thumbnail", back_populates="job", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="job", cascade="all, delete-orphan")


class Thumbnail(Base):
    """Thumbnail candidate derived from a video frame."""
    __tablename__ = "thumbnails"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    frame_index = Column(Integer)
    text_overlay = Column(Text)
    text_position = Column(String)
    style_notes = Column(Text)
    reasoning = Column(Text)
    estimated_ctr_tier = Column(String)
    file_path = Column(Text)
    generation_type = Column(String, default="pillow")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    job = relationship("Job", back_populates="thumbnails")


class Clip(Base):
    """Detected viral clip segment from a video."""
    __tablename__ = "clips"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    start_time_seconds = Column(Float)
    end_time_seconds = Column(Float)
    duration_seconds = Column(Float)
    transcript_snippet = Column(Text)
    clip_title = Column(Text)
    hook_text = Column(Text)
    virality_score = Column(Integer)
    hook_strength = Column(Integer)
    standalone_clarity = Column(Integer)
    emotional_arc = Column(Integer)
    trend_alignment = Column(Integer)
    rewatch_potential = Column(Integer)
    reasoning = Column(Text)
    suggested_caption = Column(Text)
    suggested_duration = Column(String)
    clip_type = Column(String)
    edit_suggestions = Column(Text)  # JSON string
    preview_path = Column(Text)  # Path to playable .mp4 preview (must be set before marking complete)
    kept = Column(Integer, default=0)  # 0 = not kept, 1 = kept
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    job = relationship("Job", back_populates="clips")


class Export(Base):
    """Exported clip file in a specific platform format."""
    __tablename__ = "exports"

    id = Column(String, primary_key=True)
    clip_id = Column(String, ForeignKey("clips.id"), nullable=False)
    batch_id = Column(String, ForeignKey("export_batches.id"))
    platform = Column(String)
    caption_style = Column(String)
    file_path = Column(Text)
    file_size_bytes = Column(Integer)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class ExportBatch(Base):
    """Tracks a batch export job across multiple clips."""
    __tablename__ = "export_batches"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    platform = Column(String, nullable=False)
    vertical_crop = Column(Integer, default=1)  # 1 = vertical 9:16, 0 = original
    captions = Column(Integer, default=1)  # 1 = burn-in captions, 0 = no captions
    status = Column(String, default="pending")  # pending, processing, complete, failed
    progress_message = Column(Text, default="")
    total_clips = Column(Integer, default=0)
    completed_clips = Column(Integer, default=0)
    zip_path = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Engine and session setup
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.effective_database_url,
    echo=False,
    connect_args={"check_same_thread": False},  # Required for SQLite + multi-thread (RQ workers)
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables and apply lightweight column migrations. Called on FastAPI startup."""
    Base.metadata.create_all(engine)
    _migrate_add_missing_columns()


def _migrate_add_missing_columns() -> None:
    """
    Add columns that were introduced after the initial schema but are missing from
    existing databases. SQLAlchemy's create_all() only creates missing tables,
    it does not add new columns to existing tables. This handles the gap.

    Safe to run multiple times — uses 'ALTER TABLE IF NOT EXISTS' pattern via
    exception catching (SQLite does not support IF NOT EXISTS on ALTER TABLE).
    """
    import logging
    import sqlite3

    logger = logging.getLogger(__name__)

    # Maps (table, column) -> (DDL to add the column, optional backfill SQL)
    # SQLite does not allow non-constant defaults in ALTER TABLE ADD COLUMN.
    # We add columns as nullable, then backfill from created_at or NOW().
    migrations = {
        ("jobs", "updated_at"): (
            "ALTER TABLE jobs ADD COLUMN updated_at DATETIME",
            "UPDATE jobs SET updated_at = COALESCE(created_at, datetime('now')) WHERE updated_at IS NULL",
        ),
        ("thumbnails", "updated_at"): (
            "ALTER TABLE thumbnails ADD COLUMN updated_at DATETIME",
            "UPDATE thumbnails SET updated_at = COALESCE(created_at, datetime('now')) WHERE updated_at IS NULL",
        ),
        ("clips", "updated_at"): (
            "ALTER TABLE clips ADD COLUMN updated_at DATETIME",
            "UPDATE clips SET updated_at = COALESCE(created_at, datetime('now')) WHERE updated_at IS NULL",
        ),
        ("exports", "updated_at"): (
            "ALTER TABLE exports ADD COLUMN updated_at DATETIME",
            "UPDATE exports SET updated_at = COALESCE(created_at, datetime('now')) WHERE updated_at IS NULL",
        ),
        ("clips", "kept"): (
            "ALTER TABLE clips ADD COLUMN kept INTEGER DEFAULT 0",
            None,  # 0 = not kept, correct default
        ),
        ("exports", "batch_id"): (
            "ALTER TABLE exports ADD COLUMN batch_id TEXT",
            None,
        ),
        ("jobs", "song_title"): (
            "ALTER TABLE jobs ADD COLUMN song_title TEXT",
            None,
        ),
        ("jobs", "song_artist"): (
            "ALTER TABLE jobs ADD COLUMN song_artist TEXT",
            None,
        ),
        ("jobs", "song_confidence"): (
            "ALTER TABLE jobs ADD COLUMN song_confidence TEXT",
            None,
        ),
    }

    url = settings.effective_database_url
    if not url.startswith("sqlite:///"):
        # Only handle SQLite — other databases need proper migration tooling
        return

    db_file = url.replace("sqlite:///", "")
    if db_file == ":memory:":
        return  # In-memory DBs are always fresh; no migration needed

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        for (table, column), (ddl, backfill_sql) in migrations.items():
            # Check if column already exists
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                try:
                    cursor.execute(ddl)
                    if backfill_sql:
                        cursor.execute(backfill_sql)
                    conn.commit()
                    logger.info(f"Migration: added column {table}.{column} and backfilled values")
                except sqlite3.OperationalError as exc:
                    logger.warning(f"Migration skipped {table}.{column}: {exc}")

        conn.close()
    except Exception as exc:
        logger.error(f"Schema migration failed: {exc}")


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
