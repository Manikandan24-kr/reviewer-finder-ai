"""
Main search orchestrator.

Coordinates the full reviewer-finding pipeline:
1. LLM topic extraction
2. Query embedding
3. Vector similarity search (Qdrant or in-memory numpy fallback)
4. LLM re-ranking
5. Contact enrichment
6. COI detection
"""

from __future__ import annotations

import json
import os

import numpy as np

from services.llm_service import extract_topics, rerank_candidates
from services.embedding_service import embed_query
from services.contact_service import enrich_candidates
from services.coi_service import check_all_candidates
import config

# ── In-memory vector search fallback ──────────────────────────────────────────

_inmemory_embeddings: np.ndarray | None = None
_inmemory_metadata: list[dict] | None = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_inmemory_index():
    """Load pre-computed embeddings and metadata for numpy-based search."""
    global _inmemory_embeddings, _inmemory_metadata
    if _inmemory_embeddings is not None:
        return

    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    meta_path = os.path.join(DATA_DIR, "embeddings_metadata.json")

    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"In-memory index files not found at {DATA_DIR}. "
            "Run the seed pipeline first or ensure embeddings.npy and "
            "embeddings_metadata.json exist."
        )

    _inmemory_embeddings = np.load(emb_path).astype(np.float32)
    with open(meta_path) as f:
        _inmemory_metadata = json.load(f)

    print(f"[Search] Loaded in-memory index: {_inmemory_embeddings.shape[0]} authors, "
          f"{_inmemory_embeddings.shape[1]} dims")


def _search_inmemory(
    query_vector: list[float],
    limit: int = 100,
    min_works: int = 3,
) -> list[dict]:
    """Cosine similarity search using numpy (fallback when Qdrant is unavailable)."""
    _load_inmemory_index()

    query = np.array(query_vector, dtype=np.float32)
    # Normalize query (embeddings are already normalized from sentence-transformers)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    # Cosine similarity = dot product (since both are L2-normalized)
    similarities = _inmemory_embeddings @ query

    # Get top indices
    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        meta = _inmemory_metadata[idx]
        if meta.get("works_count", 0) < min_works:
            continue
        results.append({
            "score": float(similarities[idx]),
            **meta,
        })
        if len(results) >= limit:
            break

    return results


def _is_qdrant_available() -> bool:
    """Check if Qdrant is reachable."""
    try:
        import httpx
        url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections"
        resp = httpx.get(url, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── Main pipeline ─────────────────────────────────────────────────────────────

def find_reviewers(
    title: str,
    abstract: str,
    keywords: list[str],
    author_names: list[str] | None = None,
    author_institutions: list[str] | None = None,
    num_reviewers: int = 15,
    num_vector_candidates: int = 50,
) -> dict:
    """
    End-to-end reviewer finding pipeline.

    Args:
        title: Paper title
        abstract: Paper abstract
        keywords: Paper keywords
        author_names: Names of paper authors (for COI detection)
        author_institutions: Institutions of paper authors (for COI)
        num_reviewers: Number of reviewers to return
        num_vector_candidates: How many candidates to pull from vector search
                               before LLM re-ranking (more = better accuracy, slower)

    Returns:
        Dict with extracted_topics, reviewers list, and metadata
    """
    author_names = author_names or []
    author_institutions = author_institutions or []

    steps = []

    # Step 1: LLM Topic Extraction
    steps.append("Extracting research topics with Claude...")
    try:
        topics = extract_topics(title, abstract, keywords)
    except Exception as e:
        topics = {
            "primary_domains": keywords[:3] if keywords else [],
            "methodologies": [],
            "sub_topics": keywords,
            "expanded_terms": [],
            "interdisciplinary_bridges": [],
        }
        steps.append(f"Topic extraction fallback (error: {e})")

    # Step 2: Generate query embedding
    steps.append("Generating query embedding...")
    query_vector = embed_query(
        title=title,
        abstract=abstract,
        keywords=keywords,
        extracted_topics=topics.get("primary_domains", []) + topics.get("sub_topics", []),
    )

    # Step 3: Vector similarity search
    use_qdrant = _is_qdrant_available()

    if use_qdrant:
        steps.append(f"Searching {num_vector_candidates} candidates in Qdrant...")
        from qdrant_client import QdrantClient
        from pipeline.index_qdrant import search_similar

        qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            check_compatibility=False,
        )
        candidates = search_similar(
            client=qdrant,
            collection_name=config.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=num_vector_candidates,
            min_works=3,
        )
    else:
        steps.append(f"Searching {num_vector_candidates} candidates (in-memory)...")
        candidates = _search_inmemory(
            query_vector=query_vector,
            limit=num_vector_candidates,
            min_works=3,
        )

    if not candidates:
        return {
            "extracted_topics": topics,
            "reviewers": [],
            "steps": steps + ["No candidates found in vector search."],
            "metadata": {"vector_candidates": 0, "final_reviewers": 0},
        }

    steps.append(f"Found {len(candidates)} vector candidates")

    # Step 4: LLM Re-ranking
    steps.append("Re-ranking candidates with Claude...")
    # Send top candidates to Claude for detailed scoring
    candidates_for_rerank = candidates[:min(len(candidates), 30)]  # Cap at 30 for LLM context
    try:
        scored = rerank_candidates(title, abstract, keywords, candidates_for_rerank)
    except Exception as e:
        steps.append(f"Re-ranking fallback (error: {e})")
        # Fallback: use vector scores directly
        scored = candidates_for_rerank
        for c in scored:
            c["overall_score"] = c.get("score", 0) * 10
            c["topic_score"] = c.get("score", 0) * 10
            c["methodology_score"] = 5.0
            c["seniority_score"] = min(c.get("h_index", 0) / 5, 10)
            c["recency_score"] = 5.0
            c["reasoning"] = "Ranked by vector similarity (LLM re-ranking unavailable)"

    # Step 5: Contact enrichment
    steps.append("Enriching contact information...")
    scored = enrich_candidates(scored)

    # Step 6: COI detection
    steps.append("Checking for conflicts of interest...")
    scored = check_all_candidates(
        scored,
        paper_author_names=author_names,
        paper_author_institutions=author_institutions,
    )

    # Take top N reviewers
    reviewers = scored[:num_reviewers]

    # Add rank numbers
    for i, r in enumerate(reviewers):
        r["rank"] = i + 1

    steps.append(f"Returning {len(reviewers)} reviewers")

    return {
        "extracted_topics": topics,
        "reviewers": reviewers,
        "steps": steps,
        "metadata": {
            "vector_candidates": len(candidates),
            "reranked_candidates": len(scored),
            "final_reviewers": len(reviewers),
        },
    }
