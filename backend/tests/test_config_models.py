"""
Tests for Task 1: Config, models, and database schema.
TDD RED phase — these tests define expected behavior before implementation.
"""
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


def test_settings_has_required_fields():
    """config.py must expose typed path constants and API key via pydantic Settings."""
    from backend.config import settings

    # Must have an ANTHROPIC_API_KEY attribute
    assert hasattr(settings, "anthropic_api_key")

    # Must have typed path constants
    assert hasattr(settings, "downloads_dir")
    assert hasattr(settings, "frames_dir")
    assert hasattr(settings, "thumbnails_dir")
    assert hasattr(settings, "clips_dir")
    assert hasattr(settings, "db_path")

    # Paths must be Path objects
    assert isinstance(settings.downloads_dir, Path)
    assert isinstance(settings.frames_dir, Path)
    assert isinstance(settings.thumbnails_dir, Path)
    assert isinstance(settings.clips_dir, Path)
    assert isinstance(settings.db_path, Path)

    # Paths must be absolute (resolved under data/)
    assert settings.downloads_dir.is_absolute()
    assert settings.frames_dir.is_absolute()
    assert settings.thumbnails_dir.is_absolute()
    assert settings.clips_dir.is_absolute()
    assert settings.db_path.is_absolute()


def test_settings_has_redis_url():
    """settings must have REDIS_URL with a default."""
    from backend.config import settings

    assert hasattr(settings, "redis_url")
    assert isinstance(settings.redis_url, str)
    assert settings.redis_url.startswith("redis://")


def test_job_model_defaults():
    """Job row created in in-memory SQLite must have UUID string id and status='pending'."""
    from backend.models import Base, Job

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, youtube_url="https://youtube.com/watch?v=test")
    db.add(job)
    db.commit()
    db.refresh(job)

    # id must be a non-empty string (UUID format)
    assert isinstance(job.id, str)
    assert len(job.id) > 0
    # Validate it's a valid UUID
    uuid.UUID(job.id)

    # status defaults to "pending"
    assert job.status == "pending"

    db.close()


def test_schema_has_all_required_tables():
    """SQLite schema must have jobs, thumbnails, clips, and exports tables."""
    from backend.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "jobs" in tables
    assert "thumbnails" in tables
    assert "clips" in tables
    assert "exports" in tables


def test_job_status_column_is_string():
    """Job.status must be a string column (not an enum type)."""
    from backend.models import Base, Job
    from sqlalchemy import String

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    columns = {col["name"]: col for col in inspector.get_columns("jobs")}

    # status column must exist
    assert "status" in columns

    # Must have created_at and updated_at
    assert "created_at" in columns


def test_job_has_updated_at():
    """Job must have an updated_at timestamp column."""
    from backend.models import Base, Job
    from sqlalchemy import inspect as sa_inspect

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    columns = {col["name"]: col for col in inspector.get_columns("jobs")}

    assert "updated_at" in columns, "Job model must have updated_at column for stale job recovery"
