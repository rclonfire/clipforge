#!/bin/bash
# Run the ClipForge frontend from THIS repo (clipforge-public).
cd "$(dirname "$0")/frontend"
export PATH="/Users/rondicksonjr/.nvm/versions/node/v20.20.1/bin:$PATH"
# Requires `npm install` to have been run once in this frontend/ directory.
exec npx vite --host 0.0.0.0 --port 5173
