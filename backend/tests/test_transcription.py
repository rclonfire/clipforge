"""
Tests for Task 2: Transcription service with word-level timestamps.
TDD RED phase — defines expected behavior before implementation.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_word(word: str, start: float, end: float, probability: float = 0.95):
    """Create a mock faster-whisper Word object."""
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.probability = probability
    return w


def _make_mock_segment(text: str, start: float, end: float, words=None):
    """Create a mock faster-whisper Segment object."""
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.words = words or []
    return seg


def _make_mock_info(language: str = "en", language_probability: float = 0.99, duration: float = 10.5):
    """Create a mock faster-whisper TranscriptionInfo object."""
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


# ---------------------------------------------------------------------------
# Module-level model loading
# ---------------------------------------------------------------------------

class TestModelLoadingAtModuleLevel:
    """WhisperModel must be loaded once at module import, not per-call."""

    def test_model_loaded_at_module_level_not_inside_function(self):
        """_model or equivalent module-level variable must exist after import."""
        import importlib
        import backend.services.transcription as trans_module

        # The module must expose a module-level model (not just a function)
        # Either _model directly or via a lazy-loader that caches on first call
        has_model_var = hasattr(trans_module, "_model")
        assert has_model_var, (
            "transcription.py must define _model at module level. "
            "Loading the model inside transcribe_video() is the documented OOM pattern (PITFALLS.md #5)."
        )

    def test_transcribe_video_does_not_call_whispier_model_constructor(self, tmp_path):
        """WhisperModel() constructor must NOT be called inside transcribe_video()."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        # Track WhisperModel constructor calls during transcribe_video
        constructor_call_count = [0]

        class TrackedWhisperModel:
            def __init__(self, *args, **kwargs):
                constructor_call_count[0] += 1

            def transcribe(self, *args, **kwargs):
                mock_word = _make_mock_word("hello", 0.0, 0.5)
                mock_seg = _make_mock_segment(" hello", 0.0, 0.5, words=[mock_word])
                mock_info = _make_mock_info()
                return iter([mock_seg]), mock_info

        # Create a mock model instance — this does NOT increment constructor_call_count
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = (
            iter([_make_mock_segment(" hello", 0.0, 0.5, words=[_make_mock_word("hello", 0.0, 0.5)])]),
            _make_mock_info()
        )

        # Patch the module-level _model (already loaded) with our mock instance
        # and also patch the WhisperModel class so if it IS called inside the function we catch it
        with patch("backend.services.transcription.WhisperModel", TrackedWhisperModel):
            with patch.object(trans_module, "_model", mock_model_instance):
                with patch("subprocess.run") as mock_run:
                    # Mock ffmpeg audio extraction
                    def capture_run(cmd, **kwargs):
                        if "-ar" in cmd:
                            output_path = cmd[-1]
                            Path(output_path).write_bytes(b"fake audio wav")
                        r = MagicMock()
                        r.returncode = 0
                        r.stderr = ""
                        return r
                    mock_run.side_effect = capture_run
                    trans_module.transcribe_video(str(fake_video), "test-job-trans-001")

        # WhisperModel constructor must not have been called DURING the function call
        assert constructor_call_count[0] == 0, (
            "WhisperModel() constructor was called inside transcribe_video(). "
            "The model must be loaded at module level only."
        )


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------

class TestTranscribeVideoReturnShape:
    """transcribe_video must return a dict with the correct keys and types."""

    def test_returns_dict_with_words_key(self, tmp_path):
        """transcribe_video must return a dict containing 'words' key."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hello", 0.0, 0.4)
        word2 = _make_mock_word(" world", 0.5, 0.9)
        seg = _make_mock_segment(" Hello world", 0.0, 1.0, words=[word1, word2])
        info = _make_mock_info(duration=1.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-002")

        assert "words" in result

    def test_returns_dict_with_text_key(self, tmp_path):
        """transcribe_video must return a dict containing 'text' key."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hello", 0.0, 0.4)
        seg = _make_mock_segment(" Hello", 0.0, 0.5, words=[word1])
        info = _make_mock_info(duration=0.5)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-003")

        assert "text" in result
        assert isinstance(result["text"], str)

    def test_returns_dict_with_transcript_path_key(self, tmp_path):
        """transcribe_video must return a dict containing 'transcript_path' key."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hi", 0.0, 0.3)
        seg = _make_mock_segment(" Hi", 0.0, 0.3, words=[word1])
        info = _make_mock_info(duration=0.3)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-004")

        assert "transcript_path" in result

    def test_returns_dict_with_duration_key(self, tmp_path):
        """transcribe_video must return a dict containing 'duration' key (float)."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hi", 0.0, 0.3)
        seg = _make_mock_segment(" Hi", 0.0, 0.3, words=[word1])
        info = _make_mock_info(duration=42.5)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-005")

        assert "duration" in result
        assert isinstance(result["duration"], float)

    def test_words_list_has_correct_per_word_shape(self, tmp_path):
        """Each word dict must have keys: word, start, end, probability."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hello", 0.1, 0.4, probability=0.98)
        word2 = _make_mock_word(" world", 0.5, 0.9, probability=0.95)
        seg = _make_mock_segment(" Hello world", 0.1, 0.9, words=[word1, word2])
        info = _make_mock_info(duration=1.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-006")

        words = result["words"]
        assert len(words) == 2
        for w in words:
            assert "word" in w, f"Missing 'word' key in: {w}"
            assert "start" in w, f"Missing 'start' key in: {w}"
            assert "end" in w, f"Missing 'end' key in: {w}"
            assert "probability" in w, f"Missing 'probability' key in: {w}"
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)
            assert isinstance(w["probability"], float)

    def test_words_flattened_from_multiple_segments(self, tmp_path):
        """Words from multiple segments must be flattened into a single list."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" Hello", 0.0, 0.4)
        word2 = _make_mock_word(" world", 0.5, 0.9)
        seg1 = _make_mock_segment(" Hello world", 0.0, 1.0, words=[word1, word2])

        word3 = _make_mock_word(" foo", 1.1, 1.4)
        word4 = _make_mock_word(" bar", 1.5, 1.9)
        seg2 = _make_mock_segment(" foo bar", 1.0, 2.0, words=[word3, word4])

        info = _make_mock_info(duration=2.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-007")

        # All 4 words from both segments must appear in the flat list
        assert len(result["words"]) == 4


# ---------------------------------------------------------------------------
# Transcript file written to disk
# ---------------------------------------------------------------------------

class TestTranscriptJsonWritten:
    """transcript.json must be written to data/downloads/{job_id}/transcript.json."""

    def test_transcript_json_written_to_job_dir(self, tmp_path):
        """transcript.json must exist in the job's download directory after transcription."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" test", 0.0, 0.5)
        seg = _make_mock_segment(" test", 0.0, 0.5, words=[word1])
        info = _make_mock_info(duration=0.5)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-008")

        # transcript.json must exist at the path returned in result
        transcript_path = Path(result["transcript_path"])
        assert transcript_path.exists(), f"transcript.json not found at {transcript_path}"

    def test_transcript_json_has_correct_structure(self, tmp_path):
        """transcript.json must contain 'words', 'text', and 'duration' keys."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" hello", 0.0, 0.4, probability=0.97)
        seg = _make_mock_segment(" hello", 0.0, 0.5, words=[word1])
        info = _make_mock_info(duration=0.5)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-009")

        transcript_path = Path(result["transcript_path"])
        data = json.loads(transcript_path.read_text())

        assert "words" in data, f"transcript.json missing 'words' key: {list(data.keys())}"
        assert "text" in data, f"transcript.json missing 'text' key: {list(data.keys())}"
        assert "duration" in data, f"transcript.json missing 'duration' key: {list(data.keys())}"

    def test_transcript_json_words_include_timestamps(self, tmp_path):
        """Each word entry in transcript.json must include start and end timestamps."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" awesome", 1.23, 1.78, probability=0.99)
        seg = _make_mock_segment(" awesome", 1.0, 2.0, words=[word1])
        info = _make_mock_info(duration=2.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
                (tmp_path / "audio.wav").write_bytes(b"fake")
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-010")

        transcript_path = Path(result["transcript_path"])
        data = json.loads(transcript_path.read_text())
        words = data["words"]

        assert len(words) == 1
        w = words[0]
        assert "start" in w
        assert "end" in w
        assert abs(w["start"] - 1.23) < 0.01
        assert abs(w["end"] - 1.78) < 0.01


# ---------------------------------------------------------------------------
# Audio extraction via ffmpeg subprocess
# ---------------------------------------------------------------------------

class TestAudioExtractionViaFfmpeg:
    """Audio must be extracted using ffmpeg subprocess (not moviepy or pydub)."""

    def test_ffmpeg_called_for_audio_extraction(self, tmp_path):
        """subprocess.run must be called with ffmpeg for audio extraction."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" hi", 0.0, 0.3)
        seg = _make_mock_segment(" hi", 0.0, 0.3, words=[word1])
        info = _make_mock_info(duration=0.3)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            # Create the audio file that would be created by ffmpeg
            if "-ar" in cmd:
                # This is the ffmpeg audio extraction call
                output_path = cmd[-1]  # Last arg is output path
                Path(output_path).write_bytes(b"fake audio wav")
            mock_r = MagicMock()
            mock_r.returncode = 0
            mock_r.stderr = ""
            return mock_r

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", side_effect=capture_run):
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-011")

        # At least one call must be an ffmpeg audio extraction
        ffmpeg_calls = [cmd for cmd in captured_cmds if any("ffmpeg" in str(a) for a in cmd)]
        assert len(ffmpeg_calls) >= 1, (
            f"Expected at least one ffmpeg subprocess call for audio extraction. "
            f"Got: {captured_cmds}"
        )

        # The ffmpeg command must include audio conversion flags
        ffmpeg_cmd = ffmpeg_calls[0]
        cmd_str = " ".join(str(a) for a in ffmpeg_cmd)
        assert "-ar" in cmd_str or "16000" in cmd_str, (
            f"ffmpeg command must include -ar 16000 for audio extraction. Got: {cmd_str}"
        )

    def test_audio_wav_written_to_job_dir(self, tmp_path):
        """Audio extraction must write audio.wav to data/downloads/{job_id}/."""
        import backend.services.transcription as trans_module

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        word1 = _make_mock_word(" hi", 0.0, 0.3)
        seg = _make_mock_segment(" hi", 0.0, 0.3, words=[word1])
        info = _make_mock_info(duration=0.3)

        mock_model = MagicMock()
        # We need to ensure model.transcribe is called with the audio.wav path
        transcribe_call_args = []

        def mock_transcribe(audio_path, **kwargs):
            transcribe_call_args.append(audio_path)
            return iter([seg]), info

        mock_model.transcribe = mock_transcribe

        def capture_run(cmd, **kwargs):
            # Create the audio file that ffmpeg would create
            if "-ar" in cmd:
                output_path = cmd[-1]
                Path(output_path).write_bytes(b"fake audio wav")
            mock_r = MagicMock()
            mock_r.returncode = 0
            mock_r.stderr = ""
            return mock_r

        with patch.object(trans_module, "_model", mock_model):
            with patch("subprocess.run", side_effect=capture_run):
                result = trans_module.transcribe_video(str(fake_video), "test-job-trans-012")

        # Whisper must be called with the audio.wav path (not the video)
        assert len(transcribe_call_args) > 0
        audio_path_used = transcribe_call_args[0]
        assert "audio.wav" in audio_path_used, (
            f"Whisper transcribe() must be called with audio.wav path. "
            f"Got: {audio_path_used!r}"
        )


# ---------------------------------------------------------------------------
# Function signature
# ---------------------------------------------------------------------------

class TestTranscribeVideoSignature:
    """transcribe_video must accept (video_path, job_id) — not (video_path, youtube_url)."""

    def test_transcribe_video_accepts_job_id_parameter(self):
        """transcribe_video must accept job_id as second positional parameter."""
        import inspect
        from backend.services.transcription import transcribe_video

        sig = inspect.signature(transcribe_video)
        params = list(sig.parameters.keys())
        assert "video_path" in params, "Missing 'video_path' parameter"
        assert "job_id" in params, (
            f"Missing 'job_id' parameter. Got: {params}. "
            "The plan specifies transcribe_video(video_path, job_id) — "
            "not (video_path, youtube_url)."
        )

    def test_transcribe_video_does_not_require_youtube_url(self):
        """transcribe_video must NOT have youtube_url as a required parameter."""
        import inspect
        from backend.services.transcription import transcribe_video

        sig = inspect.signature(transcribe_video)
        params = sig.parameters

        # youtube_url must not be a required parameter
        if "youtube_url" in params:
            param = params["youtube_url"]
            assert param.default is not inspect.Parameter.empty, (
                "youtube_url must have a default value (it's not in the plan's interface)"
            )


# ---------------------------------------------------------------------------
# job_processor integration
# ---------------------------------------------------------------------------

class TestJobProcessorTranscribeWired:
    """process_job must call transcribe_video (not placeholder) after download."""

    def test_process_job_calls_real_transcribe_video(self):
        """process_job must call backend.services.transcription.transcribe_video."""
        from backend.workers.job_processor import process_job
        import uuid
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.models import Base, Job

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job_id = str(uuid.uuid4())
        job = Job(id=job_id, youtube_url="https://youtube.com/watch?v=test", status="pending")
        db.add(job)
        db.commit()

        download_result = {
            "video_path": "/fake/video.mp4",
            "title": "Test Video",
            "duration_seconds": 120,
            "thumbnail_url": None,
        }
        transcribe_result = {
            "transcript_path": "/fake/transcript.json",
            "words": [{"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99}],
            "text": "hello",
            "duration": 120.0,
        }

        with patch("backend.workers.job_processor.download_video", return_value=download_result) as mock_dl:
            with patch("backend.workers.job_processor.transcribe_video", return_value=transcribe_result) as mock_tr:
                process_job(job_id=job_id, youtube_url=job.youtube_url, db_path=":memory:", _db_override=db)

        # Both must have been called
        mock_dl.assert_called_once()
        mock_tr.assert_called_once()

    def test_process_job_sets_transcribing_status(self):
        """process_job must set status='transcribing' before calling transcribe_video."""
        from backend.workers.job_processor import process_job
        import uuid
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.models import Base, Job

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job_id = str(uuid.uuid4())
        job = Job(id=job_id, youtube_url="https://youtube.com/watch?v=test", status="pending")
        db.add(job)
        db.commit()

        statuses_seen = []

        def capture_transcribe(video_path, job_id_arg, **kwargs):
            # Check what status the job has right now
            db.expire_all()
            j = db.query(Job).filter(Job.id == job_id).first()
            statuses_seen.append(j.status)
            return {
                "transcript_path": "/fake/transcript.json",
                "words": [],
                "text": "",
                "duration": 0.0,
            }

        download_result = {
            "video_path": "/fake/video.mp4",
            "title": "Test",
            "duration_seconds": 60,
            "thumbnail_url": None,
        }

        with patch("backend.workers.job_processor.download_video", return_value=download_result):
            with patch("backend.workers.job_processor.transcribe_video", side_effect=capture_transcribe):
                process_job(job_id=job_id, youtube_url=job.youtube_url, db_path=":memory:", _db_override=db)

        assert "transcribing" in statuses_seen, (
            f"Expected job status='transcribing' before transcribe_video call. "
            f"Statuses seen: {statuses_seen}"
        )
