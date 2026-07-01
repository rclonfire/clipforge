"""
Creator-style service for ClipForge.

Holds a "style profile" derived from the videos a creator actually likes and
engages with, so generated post copy reads like what already performs in their
world instead of generic AI copy. The profile is a JSON document stored at
``data/style_profile.json``:

    {
      "niche": "...",
      "updated_at": "2026-06-30T...",
      "youtube": {hook_style, title_patterns[], length, emoji, cta, hashtags[], avoid},
      "tiktok":  {hook_style, title_patterns[], length, emoji, cta, hashtags[], avoid},
      "examples": [{text, why_it_works}, ...]
    }

post_prep loads the profile and feeds style_guide_text() into the copy prompt.
derive_style_profile() builds the profile from reference videos via Claude; the
profile can also be hand-authored / edited directly.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from backend.config import settings, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

MODEL_ID = "claude-opus-4-8"

_DERIVE_SYSTEM_PROMPT = (
    "You analyze the short-form videos a specific viewer likes and engages with, then "
    "extract the repeatable patterns in their titles, captions, hooks, and hashtags so "
    "another creator can write posts in that same proven style. Work only from the "
    "supplied items — do not invent a niche the data doesn't support.\n\n"
    "Return a single JSON object and nothing else:\n"
    '{"niche": "...", '
    '"youtube": {"hook_style": "...", "title_patterns": ["..."], "length": "...", '
    '"emoji": "...", "cta": "...", "hashtags": ["#..."], "avoid": "..."}, '
    '"tiktok": {"hook_style": "...", "title_patterns": ["..."], "length": "...", '
    '"emoji": "...", "cta": "...", "hashtags": ["#..."], "avoid": "..."}, '
    '"examples": [{"text": "<a real title/caption from the data>", "why_it_works": "..."}]}'
)


def _profile_path() -> Path:
    return settings.data_dir / "style_profile.json"


def load_style_profile(path: str | Path | None = None) -> dict | None:
    """Return the stored style profile, or None if none exists / is unreadable."""
    p = Path(path) if path else _profile_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read style profile at %s: %s", p, exc)
        return None


def save_style_profile(profile: dict, path: str | Path | None = None) -> str:
    """Persist a style profile to disk and return its path."""
    p = Path(path) if path else _profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return str(p)


def style_guide_text(profile: dict | None) -> str:
    """Render a style profile into a compact prompt snippet for the copy generator."""
    if not profile:
        return ""

    lines: list[str] = []
    niche = profile.get("niche")
    if niche:
        lines.append(f"Creator niche: {niche}.")

    for platform in ("youtube", "tiktok"):
        p = profile.get(platform) or {}
        if not p:
            continue
        lines.append(f"\n{platform.upper()} — patterns that perform for the creators this viewer engages with:")
        if p.get("hook_style"):
            lines.append(f"- Hook style: {p['hook_style']}")
        if p.get("title_patterns"):
            lines.append(f"- Title/caption patterns: {'; '.join(p['title_patterns'])}")
        if p.get("length"):
            lines.append(f"- Length/format: {p['length']}")
        if p.get("emoji"):
            lines.append(f"- Emoji use: {p['emoji']}")
        if p.get("cta"):
            lines.append(f"- CTA style: {p['cta']}")
        if p.get("hashtags"):
            lines.append(f"- Favor these hashtags: {' '.join(p['hashtags'])}")
        if p.get("avoid"):
            lines.append(f"- Avoid: {p['avoid']}")

    examples = profile.get("examples") or []
    if examples:
        lines.append("\nReal examples to echo in voice (do not copy verbatim):")
        for ex in examples[:6]:
            text = ex.get("text") or ex.get("title") or ""
            why = ex.get("why_it_works") or ""
            lines.append(f'- "{text}" — {why}' if why else f'- "{text}"')

    return "\n".join(lines).strip()


def platform_hashtags(profile: dict | None, platform: str) -> list[str]:
    """Return the profile's preferred hashtags for a platform (empty if none)."""
    if not profile:
        return []
    return [str(h) for h in (profile.get(platform) or {}).get("hashtags", []) if str(h).strip()]


def derive_style_profile(reference_items: list[dict], niche: str | None = None) -> dict:
    """
    Build a style profile from reference videos (the creator's liked/engaged content) via Claude.

    Args:
        reference_items: list of dicts, e.g.
            {"platform": "youtube", "title": "...", "caption": "...",
             "hashtags": ["#..."], "channel": "...", "url": "..."}
        niche: optional hint to anchor the analysis.

    Returns the profile dict (with an 'updated_at' stamp). Requires ANTHROPIC_API_KEY.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is required to derive a style profile")
    if not reference_items:
        raise ValueError("reference_items must not be empty")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_message = (
        f"NICHE HINT (may be blank): {niche or ''}\n\n"
        f"LIKED / ENGAGED ITEMS:\n{json.dumps(reference_items, indent=2)}\n\n"
        "Extract the style profile and return the JSON object."
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=4000,
        system=_DERIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()

    profile = json.loads(text)
    profile["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if niche and not profile.get("niche"):
        profile["niche"] = niche
    return profile
