#!/bin/bash
set -e

echo "=== ClipForge Setup ==="
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "ERROR: Homebrew is required. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# Install system dependencies
echo "[1/6] Installing system dependencies..."
brew install python@3.11 node ffmpeg redis 2>/dev/null || true

# Add Python 3.11 to PATH if needed
export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"

# Verify
echo ""
echo "Checking dependencies:"
python3.11 --version || python3 --version
node --version
ffmpeg -version 2>&1 | head -1
echo ""

# Create virtual environment
echo "[2/6] Creating Python virtual environment..."
cd "$(dirname "$0")"
python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[3/6] Installing Python packages..."
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q

# Install Node dependencies
echo "[4/6] Installing frontend packages..."
cd frontend
npm install
cd ..

# Create .env file if it doesn't exist
echo "[5/6] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  >>> IMPORTANT: Add your ANTHROPIC_API_KEY to .env <<<"
else
    echo "  .env already exists"
fi

# Create data directories
echo "[6/6] Creating data directories..."
mkdir -p data/{downloads,frames,thumbnails,clips}

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Add your ANTHROPIC_API_KEY to .env"
echo "  2. Start the backend:  source .venv/bin/activate && python -m uvicorn backend.main:app --reload"
echo "  3. Start the frontend: cd frontend && npm run dev"
echo "  4. Open http://localhost:5173"
echo ""
