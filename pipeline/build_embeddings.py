"""
Generate embeddings for author profiles using sentence-transformers.

For prototype: all-MiniLM-L6-v2 (384 dims, fast, lightweight)
For production: allenai/specter2 (768 dims, trained on scientific text)
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence-transformer model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_author_text(author: dict) -> str:
    """Build a text representation of an author's expertise for embedding."""
    parts = []

    # Name and affiliation context
    if author.get("affiliations"):
        inst = author["affiliations"][0].get("institution", "")
        if inst:
            parts.append(f"Researcher at {inst}.")

    # Topics are a strong signal
    if author.get("topics"):
        parts.append(f"Research topics: {', '.join(author['topics'][:10])}.")

    # Research summary (concatenated abstracts) is the richest signal
    if author.get("research_summary"):
        # Truncate to fit model's max sequence length
        summary = author["research_summary"][:2000]
        parts.append(summary)

    return " ".join(parts)


def generate_embeddings(
    authors: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Generate embeddings for a list of author profiles."""
    texts = [build_author_text(a) for a in authors]

    # Filter out empty texts
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    valid_texts = [texts[i] for i in valid_indices]

    print(f"Generating embeddings for {len(valid_texts)} authors (batch_size={batch_size})...")
    embeddings = model.encode(
        valid_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Cosine similarity via dot product
    )

    # Map back to original indices (None for authors with no text)
    result = [None] * len(authors)
    for idx, emb in zip(valid_indices, embeddings):
        result[idx] = emb

    return result
