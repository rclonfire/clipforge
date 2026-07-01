"""
Post-prep service for ClipForge — "I prep, you post" pipeline.

Turns kept, exported clips into ready-to-post bundles for YouTube Shorts and
TikTok, plus a posting schedule. Three responsibilities:

    generate_post_copy(clips, video_title) -> {clip_id: {youtube, tiktok}}
        Platform-tailored copy. Uses Claude when ANTHROPIC_API_KEY is set,
        falling back to deterministic templates from existing clip metadata
        (so the feature works offline and stays testable).

    build_schedule(clip_ids, platforms, posts_per_day, start_date) -> [slot, ...]
        Pure function. Spreads clips across days at fixed daily slots, per
        platform. No LLM, no I/O.

    build_bundle(prep_dir, clips, copy_map, file_map, thumb_path, schedule, video_title)
        Writes per-clip folders (clip.mp4 + thumbnail + youtube.txt + tiktok.txt)
        plus schedule.md and posts.json under prep_dir.

The exported .mp4 is vertical 9:16, which is valid for both YouTube Shorts and
TikTok — the same clip file is reused for each platform.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import anthropic

from backend.config import ANTHROPIC_API_KEY, USE_PAID_APIS
from backend.prompts.post_copy_system import POST_COPY_SYSTEM_PROMPT
from backend.services.creator_style import (
    load_style_profile,
    platform_hashtags,
    style_guide_text,
)
from backend.services.song_identify import is_named

logger = logging.getLogger(__name__)

# Opus for the customer-facing copy (quality matters more than the per-call cost
# of one prep run). Clip detection uses Sonnet for its higher-volume per-chunk work.
MODEL_ID = "claude-opus-4-8"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on", "is",
    "it", "this", "that", "you", "your", "my", "with", "how", "why", "what",
    "when", "are", "was", "not", "i", "im", "me", "we", "they", "he", "she",
}


# ---------------------------------------------------------------------------
# Copy generation
# ---------------------------------------------------------------------------

def generate_post_copy(clips: list, video_title: str | None, song: dict | None = None) -> dict[str, dict]:
    """
    Return {clip_id: {"youtube": {...}, "tiktok": {...}}} for each clip.

    Conditions the copy on the creator's style profile (derived from the videos
    they like/engage with) when one exists, and on the autonomously identified
    song (song={song, artist, confidence}). When the song is named at high/medium
    confidence, captions lead with it (mixing in some vibe-only); otherwise all
    captions are vibe-only. Uses Claude when a key is configured; otherwise (or on
    any API error) falls back to deterministic templates.
    """
    profile = load_style_profile()
    named = is_named(song)

    if not USE_PAID_APIS or not ANTHROPIC_API_KEY:
        # Free/local path: deterministic templates from clip metadata + style profile.
        return {c.id: _template_copy(c, profile, song=song, named=named) for c in clips}

    try:
        copy_map = _claude_copy(clips, video_title, profile, song=song, named=named)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never block a prep
        logger.warning("Post-copy generation via Claude failed (%s) — using templates", exc)
        return {c.id: _template_copy(c, profile, song=song, named=named) for c in clips}

    # Backfill any clips the model skipped or returned malformed
    for c in clips:
        if c.id not in copy_map:
            copy_map[c.id] = _template_copy(c, profile, song=song, named=named)
    return copy_map


def _claude_copy(
    clips: list,
    video_title: str | None,
    profile: dict | None = None,
    song: dict | None = None,
    named: bool = False,
) -> dict[str, dict]:
    """Generate copy via Claude. Mirrors clip_detector's JSON-via-prompt parsing."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    payload = [
        {
            "clip_id": c.id,
            "title": c.clip_title or "",
            "hook": c.hook_text or "",
            "transcript": (c.transcript_snippet or "")[:500],
            "clip_type": c.clip_type or "",
        }
        for c in clips
    ]

    user_message = (
        f"VIDEO TITLE: {video_title or '(unknown)'}\n\n"
        f"CLIPS:\n{json.dumps(payload, indent=2)}\n\n"
        "Write YouTube Shorts and TikTok copy for every clip and return the JSON object."
    )

    guide = style_guide_text(profile)
    if guide:
        user_message += (
            "\n\nSTYLE REFERENCE — match this voice and hashtag habits "
            "(echo the patterns, do not copy examples verbatim):\n" + guide
        )

    if named and song and song.get("song"):
        artist = song.get("artist") or ""
        by = f" by {artist}" if artist else ""
        user_message += (
            f'\n\nSONG: This whole video is a violin cover of "{song["song"]}"{by}. '
            "Name the song in MOST captions (lead with it — it helps people find the cover), "
            "but make roughly one in four a vibe-only caption with no song title, for variety."
        )
    else:
        user_message += (
            "\n\nSONG: unknown — write vibe-only captions. Do not name or guess a song title."
        )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=8000,
        system=POST_COPY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    data = json.loads(text)
    result: dict[str, dict] = {}
    for entry in data.get("clips", []):
        clip_id = entry.get("clip_id")
        if not clip_id:
            continue
        yt = entry.get("youtube", {})
        tt = entry.get("tiktok", {})
        result[clip_id] = {
            "youtube": {
                "title": _ensure_shorts(str(yt.get("title", "")).strip()),
                "description": str(yt.get("description", "")).strip(),
                "tags": [str(t).strip() for t in yt.get("tags", []) if str(t).strip()],
            },
            "tiktok": {
                "caption": str(tt.get("caption", "")).strip()[:150],
                "hashtags": _clean_hashtags(tt.get("hashtags", [])),
            },
        }
    return result


def _template_copy(clip, profile: dict | None = None, song: dict | None = None, named: bool = False) -> dict:
    """Deterministic copy from a clip's existing metadata — no LLM.

    When the song is named (high/medium confidence) the caption leads with it;
    otherwise it stays vibe-only. When a creator style profile is present, its
    preferred hashtags are folded in so even offline copy carries the creator's set.
    """
    if named and song and song.get("song"):
        song_label = song["song"].strip()
        title_src = f"{song_label} on violin"
        caption = f"{song_label} on violin 🎻"
    else:
        title_src = (clip.clip_title or clip.hook_text or "Watch this").strip()
        caption = (clip.hook_text or clip.suggested_caption or clip.clip_title or "").strip()
    keywords = _keywords(clip)

    yt_profile_tags = [h.lstrip("#") for h in platform_hashtags(profile, "youtube")]
    yt_tags = _dedupe(yt_profile_tags + keywords + ["shorts"])
    yt_hashtags = " ".join(f"#{k}" for k in _dedupe(["shorts"] + yt_profile_tags + keywords[:4]))

    tt_hashtags = _clean_hashtags(
        ["#fyp"] + platform_hashtags(profile, "tiktok") + [f"#{k}" for k in keywords[:4]]
    )

    return {
        "youtube": {
            "title": _ensure_shorts(title_src),
            "description": _truncate(clip.hook_text or clip.suggested_caption or title_src, 200)
            + f"\n\n{yt_hashtags}",
            "tags": yt_tags,
        },
        "tiktok": {
            "caption": _truncate(caption, 150),
            "hashtags": tt_hashtags,
        },
    }


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _keywords(clip) -> list[str]:
    """Up to 4 topic keywords from the title, plus the clip type."""
    words = []
    source = f"{clip.clip_title or ''} {clip.hook_text or ''}".lower()
    for raw in source.split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3 and token not in _STOPWORDS and token not in words:
            words.append(token)
        if len(words) >= 4:
            break
    clip_type = (clip.clip_type or "").lower().replace(" ", "")
    if clip_type and clip_type not in words:
        words.append(clip_type)
    return words or ["clip"]


def _ensure_shorts(title: str) -> str:
    """Guarantee the YouTube title ends with #Shorts and fits in 100 chars."""
    title = title.strip()
    if "#shorts" in title.lower():
        return title[:100]
    suffix = " #Shorts"
    return title[: 100 - len(suffix)].rstrip() + suffix


def _clean_hashtags(tags) -> list[str]:
    out = []
    for t in tags:
        t = str(t).strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = "#" + t.lstrip("#")
        if t not in out:
            out.append(t)
    return out


def _truncate(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Scheduling (pure)
# ---------------------------------------------------------------------------

def build_schedule(
    clip_ids: list[str],
    platforms: list[str],
    posts_per_day: int = 3,
    start_date: date | None = None,
) -> list[dict]:
    """
    Assign each (clip, platform) pair to a posting slot.

    Clips are spread one per slot across days, independently for each platform,
    in the order given. Returns dicts sorted by time then platform:
        {clip_id, platform, scheduled_for (ISO), slot_label}
    """
    start_date = start_date or date.today()
    slots = _slot_times(posts_per_day)
    per_day = len(slots)

    schedule: list[dict] = []
    for platform in platforms:
        for idx, clip_id in enumerate(clip_ids):
            day_offset = idx // per_day
            slot = slots[idx % per_day]
            when = datetime.combine(start_date + timedelta(days=day_offset), slot)
            schedule.append(
                {
                    "clip_id": clip_id,
                    "platform": platform,
                    "scheduled_for": when.isoformat(timespec="minutes"),
                    "slot_label": _format_label(slot),
                }
            )
    return sorted(schedule, key=lambda s: (s["scheduled_for"], s["platform"]))


def _slot_times(posts_per_day: int) -> list[time]:
    presets = {
        1: [time(12, 0)],
        2: [time(11, 0), time(17, 0)],
        3: [time(11, 0), time(15, 0), time(19, 0)],
    }
    if posts_per_day in presets:
        return presets[posts_per_day]
    n = max(1, posts_per_day)
    if n == 1:
        return [time(12, 0)]
    # Even spread between 9:00 and 21:00
    start, span = 9 * 60, 12 * 60
    return [_minutes_to_time(start + round(span * i / (n - 1))) for i in range(n)]


def _minutes_to_time(total: int) -> time:
    return time(min(23, total // 60), total % 60)


def _format_label(t: time) -> str:
    hour = t.hour % 12 or 12
    ampm = "AM" if t.hour < 12 else "PM"
    return f"{hour}:{t.minute:02d} {ampm}"


# ---------------------------------------------------------------------------
# Bundle building
# ---------------------------------------------------------------------------

def build_bundle(
    prep_dir: Path,
    clips: list,
    copy_map: dict[str, dict],
    file_map: dict[str, str],
    thumb_path: str | None,
    schedule: list[dict],
    video_title: str | None,
) -> None:
    """
    Write the post-ready bundle under prep_dir (replacing any prior contents).

    Layout:
        prep_dir/
        ├── clip-01/{clip.mp4, youtube_thumbnail.jpg, youtube.txt, tiktok.txt}
        ├── clip-02/...
        ├── schedule.md
        └── posts.json
    """
    if prep_dir.exists():
        shutil.rmtree(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    posts: list[dict] = []
    for i, clip in enumerate(clips, start=1):
        folder = prep_dir / f"clip-{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        copy = copy_map.get(clip.id, _template_copy(clip))

        src = file_map.get(clip.id)
        if src and Path(src).exists():
            shutil.copy2(src, folder / "clip.mp4")
        if thumb_path and Path(thumb_path).exists():
            shutil.copy2(thumb_path, folder / "youtube_thumbnail.jpg")

        (folder / "youtube.txt").write_text(_youtube_txt(copy["youtube"]), encoding="utf-8")
        (folder / "tiktok.txt").write_text(_tiktok_txt(copy["tiktok"]), encoding="utf-8")

        posts.append(
            {
                "clip_index": i,
                "clip_id": clip.id,
                "clip_title": clip.clip_title,
                "file": "clip.mp4" if src and Path(src).exists() else None,
                "youtube": copy["youtube"],
                "tiktok": copy["tiktok"],
            }
        )

    (prep_dir / "schedule.md").write_text(
        _schedule_markdown(schedule, video_title), encoding="utf-8"
    )
    (prep_dir / "posts.json").write_text(
        json.dumps({"video_title": video_title, "posts": posts, "schedule": schedule}, indent=2),
        encoding="utf-8",
    )


def _youtube_txt(yt: dict) -> str:
    tags = ", ".join(yt.get("tags", []))
    return (
        f"TITLE:\n{yt.get('title', '')}\n\n"
        f"DESCRIPTION:\n{yt.get('description', '')}\n\n"
        f"TAGS:\n{tags}\n"
    )


def _tiktok_txt(tt: dict) -> str:
    hashtags = " ".join(tt.get("hashtags", []))
    return f"CAPTION:\n{tt.get('caption', '')}\n\nHASHTAGS:\n{hashtags}\n"


def _schedule_markdown(schedule: list[dict], video_title: str | None) -> str:
    lines = [f"# Posting schedule — {video_title or 'ClipForge clips'}", ""]
    by_day: dict[str, list[dict]] = {}
    for slot in schedule:
        day = slot["scheduled_for"][:10]
        by_day.setdefault(day, []).append(slot)

    # Stable clip labels in schedule order of first appearance
    counter = 0
    labels: dict[str, str] = {}
    for slot in sorted(schedule, key=lambda s: s["scheduled_for"]):
        if slot["clip_id"] not in labels:
            counter += 1
            labels[slot["clip_id"]] = f"clip-{counter:02d}"

    for day in sorted(by_day):
        lines.append(f"## {day}")
        lines.append("")
        lines.append("| Time | Platform | Clip |")
        lines.append("|---|---|---|")
        for slot in sorted(by_day[day], key=lambda s: (s["scheduled_for"], s["platform"])):
            lines.append(
                f"| {slot['slot_label']} | {slot['platform']} | {labels[slot['clip_id']]} |"
            )
        lines.append("")
    return "\n".join(lines)
