#!/bin/bash
set -e

echo "=== RFS Prototype Setup ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required. Install it first."
    exit 1
fi

echo "[1/4] Creating virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/4] Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  >> EDIT .env to add your ANTHROPIC_API_KEY <<"
else
    echo "  .env already exists, skipping"
fi

echo "[4/4] Checking Docker for Qdrant..."
if command -v docker &> /dev/null; then
    if docker ps --format '{{.Names}}' | grep -q qdrant; then
        echo "  Qdrant is already running"
    else
        echo "  Starting Qdrant via Docker..."
        docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
            -v qdrant_rfs_data:/qdrant/storage \
            qdrant/qdrant:v1.7.4
        echo "  Qdrant started on port 6333"
    fi
else
    echo "  WARNING: Docker not found. Install Docker and run:"
    echo "  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.7.4"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY"
echo "  2. Ensure Qdrant is running (docker ps)"
echo "  3. Seed data:      python -m pipeline.seed_prototype"
echo "  4. Launch app:     streamlit run app.py"
echo ""
