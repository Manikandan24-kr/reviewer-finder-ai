#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Seeding prototype data..."
echo "This will fetch authors from OpenAlex, generate embeddings, and index in Qdrant."
echo "Estimated time: 5-15 minutes (depending on SEED_AUTHOR_COUNT in .env)"
echo ""

python -m pipeline.seed_prototype
