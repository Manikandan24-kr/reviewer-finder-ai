"""
Embedding service — generates query vectors for search.
Wraps sentence-transformers for consistent usage across the app.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached as singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_query(title: str, abstract: str, keywords: list[str], extracted_topics: list[str] | None = None) -> list[float]:
    """Generate an embedding vector for a reviewer search query."""
    parts = [f"Title: {title}", f"Abstract: {abstract}"]

    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    if extracted_topics:
        parts.append(f"Research domains: {', '.join(extracted_topics)}")

    query_text = "\n".join(parts)

    model = get_model()
    vector = model.encode(query_text, normalize_embeddings=True)
    return vector.tolist()
