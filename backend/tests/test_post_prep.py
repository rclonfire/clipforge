"""Tests for the post-prep service (copy templates, scheduling, bundle building)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from backend.models import Clip
from backend.services import post_prep


def _clip(clip_id: str, **kw) -> Clip:
    """A detached Clip with sensible defaults; override fields via kwargs."""
    defaults = dict(
        clip_title="Why Most Diets Fail",
        hook_text="Your hormones are sabotaging your diet",
        suggested_caption="The real reason diets fail",
        transcript_snippet="Most people think willpower is the problem, but it's hormones...",
        clip_type="educational",
        start_time_seconds=0.0,
    )
    defaults.update(kw)
    return Clip(id=clip_id, **defaults)


# ---------------------------------------------------------------------------
# Copy templates
# ---------------------------------------------------------------------------

def test_template_copy_shape_and_constraints():
    copy = post_prep._template_copy(_clip("c1"))

    assert set(copy) == {"youtube", "tiktok"}
    assert copy["youtube"]["title"].endswith("#Shorts")
    assert len(copy["youtube"]["title"]) <= 100
    assert isinstance(copy["youtube"]["tags"], list) and copy["youtube"]["tags"]
    assert len(copy["tiktok"]["caption"]) <= 150
    assert all(h.startswith("#") for h in copy["tiktok"]["hashtags"])
    assert "#fyp" in copy["tiktok"]["hashtags"]


def test_ensure_shorts_is_idempotent_and_bounded():
    assert post_prep._ensure_shorts("Already tagged #Shorts") == "Already tagged #Shorts"
    long_title = "word " * 40
    out = post_prep._ensure_shorts(long_title)
    assert out.endswith("#Shorts")
    assert len(out) <= 100


def test_generate_post_copy_falls_back_to_templates_without_key(monkeypatch):
    monkeypatch.setattr(post_prep, "ANTHROPIC_API_KEY", "")
    clips = [_clip("a"), _clip("b")]

    result = post_prep.generate_post_copy(clips, "Some Video")

    assert set(result) == {"a", "b"}
    for copy in result.values():
        assert copy["youtube"]["title"].endswith("#Shorts")


def test_generate_post_copy_free_mode_skips_claude(monkeypatch):
    # Key present, but free mode -> templates only, Claude never called
    monkeypatch.setattr(post_prep, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(post_prep, "USE_PAID_APIS", False)

    def _boom(*a, **k):
        raise AssertionError("Claude must not be called in free mode")

    monkeypatch.setattr(post_prep, "_claude_copy", _boom)

    result = post_prep.generate_post_copy([_clip("a")], "Vid")
    assert "a" in result and result["a"]["youtube"]["title"].endswith("#Shorts")


def test_template_copy_names_song_when_confidently_identified():
    clip = _clip("c1")
    song = {"song": "Snooze", "artist": "SZA", "confidence": "high"}

    copy = post_prep._template_copy(clip, None, song=song, named=True)

    assert "Snooze" in copy["tiktok"]["caption"]
    assert "Snooze" in copy["youtube"]["title"]


def test_template_copy_stays_vibe_when_song_unknown():
    clip = _clip("c1")
    song = {"song": "", "artist": "", "confidence": "none"}

    copy = post_prep._template_copy(clip, None, song=song, named=False)

    # Falls back to clip metadata, not a song title
    assert copy["youtube"]["title"].endswith("#Shorts")
    assert copy["tiktok"]["caption"]  # non-empty vibe caption


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def test_build_schedule_spreads_clips_across_days():
    clip_ids = [f"c{i}" for i in range(7)]
    schedule = post_prep.build_schedule(
        clip_ids, platforms=["youtube"], posts_per_day=3, start_date=date(2026, 7, 1)
    )

    assert len(schedule) == 7
    # 3 slots/day -> clips 0,1,2 on day 1; 3,4,5 on day 2; 6 on day 3
    days = sorted({s["scheduled_for"][:10] for s in schedule})
    assert days == ["2026-07-01", "2026-07-02", "2026-07-03"]


def test_build_schedule_covers_each_platform():
    schedule = post_prep.build_schedule(
        ["c0", "c1"], platforms=["youtube", "tiktok"], posts_per_day=3, start_date=date(2026, 7, 1)
    )
    assert len(schedule) == 4  # 2 clips x 2 platforms
    assert {s["platform"] for s in schedule} == {"youtube", "tiktok"}
    # Sorted by time then platform
    times = [s["scheduled_for"] for s in schedule]
    assert times == sorted(times)


def test_build_schedule_defaults_to_today_when_no_start_date():
    schedule = post_prep.build_schedule(["c0"], platforms=["tiktok"])
    assert schedule[0]["scheduled_for"][:10] == date.today().isoformat()


# ---------------------------------------------------------------------------
# Bundle building
# ---------------------------------------------------------------------------

def test_build_bundle_writes_expected_layout(tmp_path: Path):
    clips = [_clip("a", start_time_seconds=0.0), _clip("b", start_time_seconds=30.0)]
    copy_map = {c.id: post_prep._template_copy(c) for c in clips}

    # Fake exported clip files
    src_a = tmp_path / "a.mp4"
    src_a.write_bytes(b"video-a")
    file_map = {"a": str(src_a)}  # "b" intentionally has no exported file

    schedule = post_prep.build_schedule(["a", "b"], ["youtube", "tiktok"], 3, date(2026, 7, 1))
    prep_dir = tmp_path / "postready_batch1"

    post_prep.build_bundle(prep_dir, clips, copy_map, file_map, None, schedule, "Diet Myths")

    assert (prep_dir / "clip-01" / "clip.mp4").read_bytes() == b"video-a"
    assert (prep_dir / "clip-01" / "youtube.txt").exists()
    assert (prep_dir / "clip-01" / "tiktok.txt").exists()
    # clip b had no exported file -> folder exists, copy files exist, no mp4
    assert (prep_dir / "clip-02" / "youtube.txt").exists()
    assert not (prep_dir / "clip-02" / "clip.mp4").exists()

    schedule_md = (prep_dir / "schedule.md").read_text()
    assert "Posting schedule" in schedule_md
    assert "youtube" in schedule_md and "tiktok" in schedule_md

    posts = json.loads((prep_dir / "posts.json").read_text())
    assert len(posts["posts"]) == 2
    assert posts["posts"][0]["file"] == "clip.mp4"
    assert posts["posts"][1]["file"] is None


def test_build_bundle_replaces_prior_contents(tmp_path: Path):
    prep_dir = tmp_path / "postready_batch1"
    prep_dir.mkdir()
    (prep_dir / "stale.txt").write_text("old")

    clips = [_clip("a")]
    copy_map = {"a": post_prep._template_copy(clips[0])}
    schedule = post_prep.build_schedule(["a"], ["youtube"], 3, date(2026, 7, 1))

    post_prep.build_bundle(prep_dir, clips, copy_map, {}, None, schedule, "Title")

    assert not (prep_dir / "stale.txt").exists()
    assert (prep_dir / "clip-01" / "youtube.txt").exists()
