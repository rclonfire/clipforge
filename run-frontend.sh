#!/bin/bash
cd /Users/rondicksonjr/Projects/clipforge/frontend
export PATH="/Users/rondicksonjr/.nvm/versions/node/v20.20.1/bin:$PATH"
exec npx vite --host 0.0.0.0 --port 5173
