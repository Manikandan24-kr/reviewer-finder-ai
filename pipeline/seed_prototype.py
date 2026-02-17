"""
All-in-one seed script for the prototype.

Fetches authors from OpenAlex across multiple topics, generates embeddings,
and indexes them in Qdrant. Also saves author profiles to a local JSON file
(instead of PostgreSQL) for the prototype.

Usage:
    python -m pipeline.seed_prototype
"""

from __future__ import annotations

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from pipeline.ingest_openalex import fetch_authors_by_topic, fetch_author_works, build_author_profile
from pipeline.build_embeddings import load_embedding_model, generate_embeddings
from pipeline.index_qdrant import create_collection, upload_authors
import config

# Diverse topics to get broad academic coverage
SEED_TOPICS = [
    "machine learning",
    "natural language processing",
    "computer vision",
    "climate change",
    "genomics",
    "quantum computing",
    "neuroscience",
    "renewable energy",
    "public health epidemiology",
    "materials science",
    "astrophysics",
    "economics behavioral",
    "organic chemistry",
    "robotics",
    "cybersecurity",
]

AUTHORS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "authors.json")


def main():
    target = config.SEED_AUTHOR_COUNT
    per_topic = max(target // len(SEED_TOPICS), 10)

    print(f"=== RFS Prototype Seeder ===")
    print(f"Target: ~{target} authors across {len(SEED_TOPICS)} topics")
    print(f"Per topic: {per_topic}")
    print()

    # Step 1: Fetch authors from OpenAlex
    print("STEP 1: Fetching authors from OpenAlex...")
    all_authors_raw = {}
    for topic in SEED_TOPICS:
        try:
            raw_authors = fetch_authors_by_topic(
                topic=topic,
                count=per_topic,
                email=config.OPENALEX_EMAIL,
            )
            for a in raw_authors:
                aid = a.get("id", "")
                if aid and aid not in all_authors_raw:
                    all_authors_raw[aid] = a
            print(f"  [{topic}] fetched {len(raw_authors)}, total unique: {len(all_authors_raw)}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  [{topic}] ERROR: {e}")
            continue

    print(f"\nTotal unique authors fetched: {len(all_authors_raw)}")

    # Step 2: Build detailed profiles (fetch works for each author)
    print("\nSTEP 2: Building author profiles (fetching works)...")
    profiles = []
    for i, (aid, raw) in enumerate(all_authors_raw.items()):
        try:
            works = fetch_author_works(aid, limit=5, email=config.OPENALEX_EMAIL)
            profile = build_author_profile(raw, works)
            if profile["research_summary"]:  # Only keep authors with abstracts
                profiles.append(profile)
        except Exception as e:
            print(f"  Error building profile for {aid}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_authors_raw)} authors ({len(profiles)} with abstracts)")

        time.sleep(0.05)  # Rate limiting

    print(f"\nProfiles with research summaries: {len(profiles)}")

    # Step 3: Save profiles to JSON (our prototype "database")
    os.makedirs(os.path.dirname(AUTHORS_FILE), exist_ok=True)
    with open(AUTHORS_FILE, "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"\nSTEP 3: Saved {len(profiles)} profiles to {AUTHORS_FILE}")

    # Step 4: Generate embeddings
    print(f"\nSTEP 4: Generating embeddings with {config.EMBEDDING_MODEL}...")
    model = load_embedding_model(config.EMBEDDING_MODEL)
    embeddings = generate_embeddings(profiles, model)
    valid_count = sum(1 for e in embeddings if e is not None)
    print(f"Generated {valid_count} embeddings")

    # Step 5: Upload to Qdrant
    print(f"\nSTEP 5: Indexing in Qdrant ({config.QDRANT_HOST}:{config.QDRANT_PORT})...")
    qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, check_compatibility=False)
    create_collection(qdrant, config.QDRANT_COLLECTION, config.EMBEDDING_DIMENSION)
    indexed = upload_authors(qdrant, config.QDRANT_COLLECTION, profiles, embeddings)

    print(f"\n=== SEEDING COMPLETE ===")
    print(f"  Authors indexed: {indexed}")
    print(f"  Profiles saved:  {AUTHORS_FILE}")
    print(f"  Qdrant collection: {config.QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()
