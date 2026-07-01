"""Tests for the creator-style profile (load/save, rendering, copy integration)."""
from __future__ import annotations

from backend.models import Clip
from backend.services import creator_style, post_prep


_PROFILE = {
    "niche": "natural hair / faith",
    "youtube": {
        "hook_style": "question + payoff",
        "title_patterns": ["POV: ...", "The truth about ..."],
        "hashtags": ["#naturalhair", "#faith"],
        "avoid": "clickbait that overpromises",
    },
    "tiktok": {
        "hook_style": "first 1s visual hook",
        "hashtags": ["#naturalhairtok", "#christiantiktok"],
    },
    "examples": [{"text": "POV: your wash day finally makes sense", "why_it_works": "relatable POV hook"}],
}


def test_load_save_roundtrip(tmp_path):
    path = tmp_path / "style_profile.json"
    creator_style.save_style_profile(_PROFILE, path=path)
    loaded = creator_style.load_style_profile(path=path)
    assert loaded["niche"] == "natural hair / faith"
    assert loaded["tiktok"]["hashtags"] == ["#naturalhairtok", "#christiantiktok"]


def test_load_missing_returns_none(tmp_path):
    assert creator_style.load_style_profile(path=tmp_path / "nope.json") is None


def test_style_guide_text_renders_sections():
    guide = creator_style.style_guide_text(_PROFILE)
    assert "natural hair / faith" in guide
    assert "YOUTUBE" in guide and "TIKTOK" in guide
    assert "#naturalhair" in guide
    assert "POV: your wash day finally makes sense" in guide


def test_style_guide_text_empty_for_none():
    assert creator_style.style_guide_text(None) == ""


def test_platform_hashtags():
    assert creator_style.platform_hashtags(_PROFILE, "tiktok") == ["#naturalhairtok", "#christiantiktok"]
    assert creator_style.platform_hashtags(None, "youtube") == []


def test_template_copy_folds_in_profile_hashtags():
    clip = Clip(
        id="c1",
        clip_title="Why your wash day fails",
        hook_text="You're skipping the one step that matters",
        suggested_caption="wash day tips",
        clip_type="educational",
    )
    copy = post_prep._template_copy(clip, _PROFILE)

    # Profile hashtags get folded into both platforms
    assert "#naturalhairtok" in copy["tiktok"]["hashtags"]
    assert "#fyp" in copy["tiktok"]["hashtags"]
    assert "naturalhair" in copy["youtube"]["tags"]
    assert copy["youtube"]["title"].endswith("#Shorts")
