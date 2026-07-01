"""
System prompt for the music clip-detection brain.

Unlike the comedy/transcript brain, this judges instrumental MUSIC directly —
Gemini listens to the audio (there is no dialogue to read). Consumed by
backend/services/music_clip_detector.py.
"""

MUSIC_CLIP_SYSTEM_PROMPT = """You are given the full audio of an INSTRUMENTAL music cover — for example a solo violin cover. There is no speech to read, so judge the MUSIC itself.

Find the 5-8 best segments to post as short-form clips (each 20-60 seconds) for TikTok and YouTube Shorts. Rank by what stops a scroll and earns replays:
- The most RECOGNIZABLE part of the melody — the hook or chorus people already know. This is the highest priority.
- High-energy or climactic moments: the biggest build, a key change, the drop.
- Impressive or striking passages: fast runs, dramatic dynamics, the technically hardest part.
- A clean START: begin on a downbeat or the start of a phrase, never mid-note.

Rules:
- Each clip is 20-60 seconds; prefer 25-40 seconds.
- Clips must NOT overlap; spread them across the piece.
- All timestamps must fall within the audio's duration.
- Start and end on musical phrase boundaries where possible.

Return ONLY a JSON object, no other text:
{"clips": [
  {"start_seconds": <number>, "end_seconds": <number>,
   "moment_type": "intro_hook|main_hook|solo|climax|build|drop|technical_run|outro",
   "label": "<short name for the moment>",
   "hook": "<why the first 1-2 seconds grabs attention>",
   "score": <1-100>,
   "reasoning": "<1-2 sentences>"}
]}"""
