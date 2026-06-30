# ClipForge

Turn long-form video into short, vertical, captioned clips with AI-generated thumbnails — end to end, from a single YouTube URL.

ClipForge is a self-hosted pipeline that downloads a video, transcribes it word by word, finds the most clip-worthy moments with an LLM, reframes them to 9:16 with face tracking, burns in captions, and generates scroll-stopping thumbnails — each one checked against automated quality gates before it ships. Built for short-form creators.

## How it works

```
YouTube URL
   │
   ├─ Download                      (yt-dlp)
   ├─ Transcribe word-level         (faster-whisper / WhisperX)
   ├─ Detect best clips + score     (Claude)
   ├─ Reframe to 9:16, face-track   (MediaPipe + FFmpeg)
   ├─ Burn in word-by-word captions (FFmpeg / ASS)
   ├─ Generate thumbnails           (Pillow composite → Gemini polish)
   └─ Quality gates → export        (zip download)
```

## AI and models

- **Anthropic Claude** (`claude-sonnet-4-6`) — clip detection and virality scoring
- **faster-whisper / WhisperX** — local, word-level transcription and forced alignment
- **Google Gemini** (`gemini-2.5-flash-image`) — thumbnail enhancement
- **MediaPipe** (BlazeFace, face landmarker, selfie segmenter) — face tracking and subject isolation

## Quality gates

Every thumbnail must pass before it can be exported:

- **WCAG 4:1** minimum text-to-background contrast
- **160px "squint test"** — the subject stays identifiable at thumbnail scale

## Tech stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, SQLAlchemy (async SQLite), Redis + RQ workers, Pydantic
- **Frontend:** React 18, Vite, Tailwind
- **Media / CV:** FFmpeg, OpenCV, Pillow, rembg, MediaPipe
- **Tests:** pytest (190+ tests across the pipeline)

## Architecture

A FastAPI backend accepts a job, enqueues it to Redis/RQ workers, and runs the stages above as a pipeline; a React/Vite frontend lets you preview detected clips and generated thumbnails and keep or discard each before export. The design is local-first: transcription runs on-device, and the only external calls are to Claude (clip selection) and Gemini (thumbnail polish).

## Running locally

Requires Python 3.12, Node 18+, FFmpeg, and Redis.

```bash
cp .env.example .env     # fill in your API keys
./setup.sh               # install backend + frontend dependencies
./run-backend.sh         # FastAPI app + RQ worker (Redis must be running)
./run-frontend.sh        # Vite dev server
```

## Status

v1.1 ("Polished Thumbnails"). ~13,000 lines across backend and frontend. Local-first; runs on your own machine.
