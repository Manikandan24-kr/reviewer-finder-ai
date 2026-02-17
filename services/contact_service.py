"""
Contact enrichment service.
Looks up reviewer contact information from ORCID and OpenAlex.
"""

from __future__ import annotations

import httpx
import json
import os

AUTHORS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "authors.json")

_authors_cache: dict | None = None
_authors_mtime: float = 0


def _load_authors_db() -> dict:
    """Load the local authors JSON as a lookup dict (reloads if file changed)."""
    global _authors_cache, _authors_mtime
    try:
        mtime = os.path.getmtime(AUTHORS_FILE) if os.path.exists(AUTHORS_FILE) else 0
    except OSError:
        mtime = 0
    if _authors_cache is None or mtime != _authors_mtime:
        if os.path.exists(AUTHORS_FILE):
            with open(AUTHORS_FILE) as f:
                authors = json.load(f)
            _authors_cache = {a["id"]: a for a in authors}
            _authors_mtime = mtime
        else:
            _authors_cache = {}
    return _authors_cache


def enrich_contact(candidate: dict) -> dict:
    """Enrich a candidate with contact information from local DB + ORCID."""
    author_id = candidate.get("author_id", "")
    contact = dict(candidate.get("contact", {}))

    # Step 1: Check local authors DB
    db = _load_authors_db()
    if author_id in db:
        stored = db[author_id]
        stored_contact = stored.get("contact", {})
        contact.update({k: v for k, v in stored_contact.items() if v})

        # Add co-author info for COI checking
        candidate["co_author_ids"] = stored.get("co_author_ids", [])
        candidate["affiliations"] = stored.get("affiliations", [])

    # Step 2: Try OpenAlex author API for extra contact info
    if author_id and "email" not in contact:
        openalex_contact = _fetch_openalex_contact(author_id)
        contact.update({k: v for k, v in openalex_contact.items() if v and k not in contact})

    # Step 3: Try ORCID if we have an ORCID ID
    orcid_id = contact.get("orcid") or candidate.get("orcid", "")
    if orcid_id and "email" not in contact:
        orcid_contact = _fetch_orcid_contact(orcid_id)
        contact.update({k: v for k, v in orcid_contact.items() if v})

    # Step 4: Build OpenAlex profile link
    if author_id:
        openalex_id = author_id.replace("https://openalex.org/", "")
        contact["openalex_url"] = f"https://openalex.org/{openalex_id}"

    candidate["contact"] = contact
    return candidate


def _fetch_openalex_contact(author_id: str) -> dict:
    """Fetch contact info from OpenAlex author endpoint (homepage, institution page)."""
    try:
        openalex_id = author_id.replace("https://openalex.org/", "")
        url = f"https://api.openalex.org/authors/{openalex_id}"
        resp = httpx.get(url, params={"select": "id,display_name,last_known_institutions,orcid"}, timeout=10)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        contact = {}

        # Check for institutional homepage
        for inst in (data.get("last_known_institutions") or []):
            ror = inst.get("ror", "")
            if ror:
                # ROR pages often link to institutional directories
                contact.setdefault("institution_page", ror)

        # If we got an ORCID we didn't have before
        orcid = data.get("orcid", "")
        if orcid:
            orcid_id = orcid.replace("https://orcid.org/", "")
            contact["orcid"] = orcid_id
            contact["orcid_url"] = f"https://orcid.org/{orcid_id}"

        return contact
    except Exception:
        return {}


def _fetch_orcid_contact(orcid_id: str) -> dict:
    """Fetch public contact info from ORCID API."""
    try:
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        headers = {"Accept": "application/json"}
        resp = httpx.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return {}

        data = resp.json()
        contact = {}

        # Extract email
        emails = data.get("emails", {}).get("email", [])
        for email_entry in emails:
            email = email_entry.get("email")
            if email:
                contact["email"] = email
                break

        # Extract URLs (homepage, etc.)
        urls = data.get("researcher-urls", {}).get("researcher-url", [])
        for url_entry in urls:
            url_name = (url_entry.get("url-name") or "").lower()
            url_value = (url_entry.get("url") or {}).get("value", "")
            if url_value:
                if "google scholar" in url_name or "scholar.google" in url_value:
                    contact["google_scholar"] = url_value
                elif "homepage" in url_name or "personal" in url_name:
                    contact["homepage"] = url_value
                elif not contact.get("homepage"):
                    contact["homepage"] = url_value

        return contact

    except Exception:
        return {}


def enrich_candidates(candidates: list[dict]) -> list[dict]:
    """Enrich all candidates with contact info."""
    enriched = []
    for c in candidates:
        enriched.append(enrich_contact(c))
    return enriched
