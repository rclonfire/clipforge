#!/bin/bash
cd /Users/rondicksonjr/Projects/clipforge
export PATH="/Users/rondicksonjr/Projects/clipforge/.venv/bin:$PATH"
exec /Users/rondicksonjr/Projects/clipforge/.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
