"""
Upload author embeddings and metadata to Qdrant vector database.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
import uuid


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
):
    """Create or recreate the Qdrant collection."""
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists. Recreating...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{collection_name}' (dim={vector_size}, cosine)")


def upload_authors(
    client: QdrantClient,
    collection_name: str,
    authors: list[dict],
    embeddings: list,
    batch_size: int = 100,
):
    """Upload author vectors + metadata to Qdrant."""
    points = []

    for author, embedding in zip(authors, embeddings):
        if embedding is None:
            continue

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, author["id"]))

        payload = {
            "author_id": author["id"],
            "name": author["name"],
            "orcid": author.get("orcid", ""),
            "institution": (
                author["affiliations"][0]["institution"]
                if author.get("affiliations")
                else ""
            ),
            "country": (
                author["affiliations"][0].get("country", "")
                if author.get("affiliations")
                else ""
            ),
            "topics": author.get("topics", []),
            "h_index": author.get("h_index", 0),
            "citation_count": author.get("citation_count", 0),
            "works_count": author.get("works_count", 0),
            "last_publication_date": author.get("last_publication_date", ""),
        }

        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        ))

    # Upload in batches
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)
        print(f"  Uploaded {total}/{len(points)} points")

    print(f"Done. {len(points)} authors indexed in Qdrant.")
    return len(points)


def search_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 100,
    min_works: int = 3,
    exclude_author_ids: list[str] | None = None,
) -> list[dict]:
    """Search for similar authors by vector using Qdrant REST API directly."""
    import httpx

    # Build filter
    must = [{"key": "works_count", "range": {"gte": min_works}}]
    must_not = []
    if exclude_author_ids:
        for aid in exclude_author_ids:
            must_not.append({"key": "author_id", "match": {"value": aid}})

    body = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "filter": {
            "must": must,
            **({"must_not": must_not} if must_not else {}),
        },
    }

    # Extract host/port from client config
    host = client._client.rest_uri if hasattr(client, '_client') else f"http://localhost:6333"
    url = f"http://{host.replace('http://', '').rstrip('/')}/collections/{collection_name}/points/search"
    # Fallback: build URL from known config
    try:
        resp = httpx.post(url, json=body, timeout=30)
    except Exception:
        import config
        url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections/{collection_name}/points/search"
        resp = httpx.post(url, json=body, timeout=30)

    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "score": point["score"],
            **point["payload"],
        }
        for point in data.get("result", [])
    ]
