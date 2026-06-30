"""
Unit tests for export_encoder.py.

subprocess.run is mocked to capture FFmpeg commands without executing them.
_detect_face_center and _get_video_dimensions are mocked to avoid real video files.
"""
from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from backend.services.export_encoder import PLATFORM_PRESETS, export_clip


# Shared mock targets
_SUBPROCESS_RUN = "backend.services.export_encoder.subprocess.run"
_DETECT_FACE = "backend.services.export_encoder._detect_face_center"
_GET_DIMS = "backend.services.export_encoder._get_video_dimensions"


def _run_export_clip(
    vertical_crop=True,
    ass_path=None,
    platform="tiktok",
    start=5.0,
    end=30.0,
    face_center=(0.5, 0.4),
    dimensions=(1920, 1080),
):
    """Helper: run export_clip with mocks and return the captured FFmpeg command list."""
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    with patch(_SUBPROCESS_RUN, mock_run), \
         patch(_DETECT_FACE, return_value=face_center), \
         patch(_GET_DIMS, return_value=dimensions):
        export_clip(
            video_path="/tmp/test.mp4",
            start_seconds=start,
            end_seconds=end,
            output_path="/tmp/out.mp4",
            ass_path=ass_path,
            vertical_crop=vertical_crop,
            platform=platform,
        )

    # Extract the command list from the first call's positional args
    call_args = mock_run.call_args
    cmd = call_args[0][0]  # First positional arg = cmd list
    return cmd


class TestPlatformPresets(unittest.TestCase):
    """PLATFORM_PRESETS dict has keys tiktok, shorts, original with max_duration and resolution."""

    def test_presets_has_tiktok(self):
        self.assertIn("tiktok", PLATFORM_PRESETS)

    def test_presets_has_shorts(self):
        self.assertIn("shorts", PLATFORM_PRESETS)

    def test_presets_has_original(self):
        self.assertIn("original", PLATFORM_PRESETS)

    def test_tiktok_has_max_duration(self):
        self.assertIn("max_duration", PLATFORM_PRESETS["tiktok"])

    def test_shorts_has_max_duration(self):
        self.assertIn("max_duration", PLATFORM_PRESETS["shorts"])

    def test_original_has_max_duration(self):
        self.assertIn("max_duration", PLATFORM_PRESETS["original"])

    def test_tiktok_has_resolution(self):
        p = PLATFORM_PRESETS["tiktok"]
        self.assertIn("width", p)
        self.assertIn("height", p)

    def test_tiktok_max_duration_is_60(self):
        self.assertEqual(PLATFORM_PRESETS["tiktok"]["max_duration"], 60)

    def test_shorts_max_duration_is_60(self):
        self.assertEqual(PLATFORM_PRESETS["shorts"]["max_duration"], 60)

    def test_original_max_duration_is_300(self):
        self.assertEqual(PLATFORM_PRESETS["original"]["max_duration"], 300)

    def test_tiktok_resolution_is_vertical(self):
        p = PLATFORM_PRESETS["tiktok"]
        self.assertEqual(p["width"], 1080)
        self.assertEqual(p["height"], 1920)


class TestExportClipVerticalCrop(unittest.TestCase):
    """export_clip with vertical_crop=True includes crop= in the FFmpeg -vf argument."""

    def _get_vf(self, cmd):
        idx = cmd.index("-vf")
        return cmd[idx + 1]

    def test_vertical_crop_true_includes_crop_filter(self):
        cmd = _run_export_clip(vertical_crop=True, platform="tiktok")
        vf = self._get_vf(cmd)
        self.assertIn("crop=", vf)

    def test_vertical_crop_false_does_not_include_crop_filter(self):
        cmd = _run_export_clip(vertical_crop=False, platform="original")
        vf = self._get_vf(cmd)
        self.assertNotIn("crop=", vf)

    def test_ass_path_included_in_vf(self):
        cmd = _run_export_clip(vertical_crop=True, ass_path="/tmp/test.ass", platform="tiktok")
        vf = self._get_vf(cmd)
        self.assertIn("ass=", vf)

    def test_no_ass_path_not_in_vf(self):
        cmd = _run_export_clip(vertical_crop=True, ass_path=None, platform="tiktok")
        vf = self._get_vf(cmd)
        self.assertNotIn("ass=", vf)


class TestExportClipEncoderSettings(unittest.TestCase):
    """FFmpeg command includes required encoder flags."""

    def test_includes_crf_18(self):
        cmd = _run_export_clip()
        self.assertIn("-crf", cmd)
        crf_idx = cmd.index("-crf")
        self.assertEqual(cmd[crf_idx + 1], "18")

    def test_includes_profile_high(self):
        cmd = _run_export_clip()
        self.assertIn("-profile:v", cmd)
        prof_idx = cmd.index("-profile:v")
        self.assertEqual(cmd[prof_idx + 1], "high")

    def test_includes_movflags_faststart(self):
        cmd = _run_export_clip()
        self.assertIn("-movflags", cmd)
        mf_idx = cmd.index("-movflags")
        self.assertEqual(cmd[mf_idx + 1], "+faststart")

    def test_includes_libx264_codec(self):
        cmd = _run_export_clip()
        self.assertIn("-c:v", cmd)
        cv_idx = cmd.index("-c:v")
        self.assertEqual(cmd[cv_idx + 1], "libx264")


class TestExportClipDurationClamping(unittest.TestCase):
    """export_clip clamps duration to platform max."""

    def _get_duration(self, cmd):
        idx = cmd.index("-t")
        return float(cmd[idx + 1])

    def test_tiktok_clamps_to_60s_max(self):
        # start=0, end=90 -> duration=90, clamped to 60
        cmd = _run_export_clip(platform="tiktok", start=0.0, end=90.0)
        self.assertLessEqual(self._get_duration(cmd), 60.0)

    def test_shorts_clamps_to_60s_max(self):
        cmd = _run_export_clip(platform="shorts", start=0.0, end=90.0)
        self.assertLessEqual(self._get_duration(cmd), 60.0)

    def test_original_allows_up_to_300s(self):
        # duration=120 < 300 — should not be clamped
        cmd = _run_export_clip(platform="original", start=0.0, end=120.0, vertical_crop=False)
        self.assertAlmostEqual(self._get_duration(cmd), 120.0)

    def test_short_clip_not_clamped(self):
        # duration=30 < 60 — should not be clamped
        cmd = _run_export_clip(platform="tiktok", start=0.0, end=30.0)
        self.assertAlmostEqual(self._get_duration(cmd), 30.0)


class TestExportClipOriginalPreset(unittest.TestCase):
    """export_clip for original preset normalizes to 1920x1080 (scale only, no crop)."""

    def _get_vf(self, cmd):
        idx = cmd.index("-vf")
        return cmd[idx + 1]

    def test_original_preset_no_crop(self):
        cmd = _run_export_clip(vertical_crop=False, platform="original")
        vf = self._get_vf(cmd)
        self.assertNotIn("crop=", vf)

    def test_original_preset_scale_present(self):
        cmd = _run_export_clip(vertical_crop=False, platform="original")
        vf = self._get_vf(cmd)
        self.assertIn("scale=", vf)

    def test_original_scale_is_1920x1080(self):
        cmd = _run_export_clip(vertical_crop=False, platform="original")
        vf = self._get_vf(cmd)
        self.assertIn("1920", vf)
        self.assertIn("1080", vf)


class TestExportClipReturnValue(unittest.TestCase):
    """export_clip returns output_path string on success."""

    def test_returns_output_path(self):
        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch(_SUBPROCESS_RUN, mock_run), \
             patch(_DETECT_FACE, return_value=(0.5, 0.4)), \
             patch(_GET_DIMS, return_value=(1920, 1080)):
            result = export_clip(
                video_path="/tmp/test.mp4",
                start_seconds=5.0,
                end_seconds=30.0,
                output_path="/tmp/out.mp4",
                platform="tiktok",
            )
        self.assertEqual(result, "/tmp/out.mp4")


if __name__ == "__main__":
    unittest.main()
