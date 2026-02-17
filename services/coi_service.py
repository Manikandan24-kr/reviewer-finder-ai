"""
Conflict of Interest (COI) detection service.

Checks for:
1. Co-authorship (shared publications)
2. Same institution
3. Known advisor/student relationships (heuristic)
"""

from __future__ import annotations


def detect_conflicts(
    candidate: dict,
    paper_author_names: list[str],
    paper_author_institutions: list[str],
    paper_author_ids: list[str] | None = None,
) -> list[dict]:
    """
    Detect potential conflicts of interest between a candidate reviewer
    and the paper's authors.

    Returns a list of COI flags: [{"type": "...", "detail": "..."}]
    """
    flags = []

    # 1. Co-authorship check
    co_author_ids = set(candidate.get("co_author_ids", []))
    if paper_author_ids:
        for aid in paper_author_ids:
            if aid in co_author_ids:
                flags.append({
                    "type": "co_author",
                    "detail": f"Has co-authored with paper author (ID: {aid})",
                    "severity": "high",
                })

    # 2. Same institution check
    candidate_institutions = set()
    for aff in candidate.get("affiliations", []):
        inst = aff.get("institution", "").lower().strip()
        if inst:
            candidate_institutions.add(inst)

    candidate_inst_from_payload = candidate.get("institution", "").lower().strip()
    if candidate_inst_from_payload:
        candidate_institutions.add(candidate_inst_from_payload)

    for paper_inst in paper_author_institutions:
        paper_inst_lower = paper_inst.lower().strip()
        for cand_inst in candidate_institutions:
            if paper_inst_lower and cand_inst and (
                paper_inst_lower in cand_inst or cand_inst in paper_inst_lower
            ):
                flags.append({
                    "type": "same_institution",
                    "detail": f"Same institution: {cand_inst}",
                    "severity": "medium",
                })
                break

    # 3. Name similarity check (basic — catches same person or close collaborator)
    candidate_name = candidate.get("name", "").lower().strip()
    for author_name in paper_author_names:
        author_lower = author_name.lower().strip()
        if author_lower and candidate_name:
            # Exact match
            if author_lower == candidate_name:
                flags.append({
                    "type": "same_person",
                    "detail": f"Candidate name matches paper author: {author_name}",
                    "severity": "critical",
                })
            # Last name match (rough heuristic)
            elif (
                author_lower.split()[-1] == candidate_name.split()[-1]
                and len(candidate_name.split()[-1]) > 2
            ):
                flags.append({
                    "type": "possible_relation",
                    "detail": f"Shares last name with paper author: {author_name}",
                    "severity": "low",
                })

    return flags


def check_all_candidates(
    candidates: list[dict],
    paper_author_names: list[str],
    paper_author_institutions: list[str],
    paper_author_ids: list[str] | None = None,
) -> list[dict]:
    """Run COI checks on all candidates and attach flags."""
    for candidate in candidates:
        flags = detect_conflicts(
            candidate,
            paper_author_names,
            paper_author_institutions,
            paper_author_ids,
        )
        candidate["coi_flags"] = flags
    return candidates
