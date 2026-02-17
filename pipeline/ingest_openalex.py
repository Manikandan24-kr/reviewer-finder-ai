"""
Fetch author data from OpenAlex API.

OpenAlex provides free, open access to ~90M+ scholarly works and ~70M+ authors.
We fetch authors with their works, topics, and co-author relationships.
"""

from __future__ import annotations

import httpx
import time
from tqdm import tqdm

BASE_URL = "https://api.openalex.org"


def fetch_authors_by_topic(
    topic: str,
    count: int = 100,
    email: str = "",
) -> list[dict]:
    """Fetch authors who have published on a given topic."""
    params = {
        "filter": f"default.search:{topic}",
        "sort": "cited_by_count:desc",
        "per_page": min(count, 50),
        "select": "id,display_name,orcid,last_known_institutions,topics,summary_stats,works_count,cited_by_count,works_api_url",
    }
    if email:
        params["mailto"] = email

    authors = []
    cursor = "*"

    with tqdm(total=count, desc=f"Fetching authors for '{topic}'") as pbar:
        while len(authors) < count:
            params["cursor"] = cursor
            resp = httpx.get(f"{BASE_URL}/authors", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            authors.extend(results)
            pbar.update(len(results))

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            # Polite rate limiting
            time.sleep(0.1)

    return authors[:count]


def fetch_author_works(author_id: str, limit: int = 10, email: str = "") -> list[dict]:
    """Fetch recent works (with abstracts) for an author."""
    openalex_id = author_id.replace("https://openalex.org/", "")
    params = {
        "filter": f"author.id:{openalex_id}",
        "sort": "publication_date:desc",
        "per_page": limit,
        "select": "id,title,abstract_inverted_index,publication_date,topics,authorships",
    }
    if email:
        params["mailto"] = email

    resp = httpx.get(f"{BASE_URL}/works", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as inverted indexes — reconstruct to plain text."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


def build_author_profile(raw_author: dict, works: list[dict]) -> dict:
    """Build a structured author profile from OpenAlex data."""
    # Extract institution info
    affiliations = []
    for inst in (raw_author.get("last_known_institutions") or []):
        affiliations.append({
            "institution": inst.get("display_name", "Unknown"),
            "country": inst.get("country_code", ""),
            "type": inst.get("type", ""),
            "ror": inst.get("ror", ""),
        })

    # Extract topics
    topics = []
    for t in (raw_author.get("topics") or [])[:15]:
        topics.append(t.get("display_name", ""))

    # Build research summary from recent abstracts
    abstracts = []
    co_author_ids = set()
    last_pub_date = None

    for work in works:
        abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))
        if abstract_text:
            abstracts.append(abstract_text)
        if work.get("publication_date") and not last_pub_date:
            last_pub_date = work["publication_date"]
        # Collect co-authors
        for authorship in (work.get("authorships") or []):
            aid = (authorship.get("author") or {}).get("id", "")
            if aid and aid != raw_author.get("id"):
                co_author_ids.add(aid)

    research_summary = " ".join(abstracts[:10])  # Concat top 10 abstracts

    summary_stats = raw_author.get("summary_stats") or {}

    # Build contact info from available data
    orcid = raw_author.get("orcid") or ""
    contact = {}
    if orcid:
        orcid_id = orcid.replace("https://orcid.org/", "")
        contact["orcid"] = orcid_id
        contact["orcid_url"] = f"https://orcid.org/{orcid_id}"

    if affiliations:
        contact["institution"] = affiliations[0]["institution"]

    return {
        "id": raw_author.get("id", ""),
        "name": raw_author.get("display_name", "Unknown"),
        "orcid": orcid.replace("https://orcid.org/", "") if orcid else "",
        "affiliations": affiliations,
        "topics": [t for t in topics if t],
        "h_index": summary_stats.get("h_index", 0),
        "citation_count": raw_author.get("cited_by_count", 0),
        "works_count": raw_author.get("works_count", 0),
        "last_publication_date": last_pub_date,
        "co_author_ids": list(co_author_ids)[:50],
        "contact": contact,
        "research_summary": research_summary[:5000],  # Cap at 5000 chars
    }
