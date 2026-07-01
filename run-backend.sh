#!/bin/bash
# Run the ClipForge backend from THIS repo (clipforge-public), wherever it lives.
cd "$(dirname "$0")"

# Prefer a local .venv; fall back to the shared clipforge venv where deps are installed.
if [ -x ".venv/bin/python" ]; then
  VENV_PY="$(pwd)/.venv/bin/python"
else
  VENV_PY="/Users/rondicksonjr/Projects/clipforge/.venv/bin/python"
fi

export PATH="$(dirname "$VENV_PY"):$PATH"
exec "$VENV_PY" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
