"""
LLM service — uses Claude for topic extraction and candidate re-ranking.

When ANTHROPIC_API_KEY is not set, falls back to smart heuristic-based
mock implementations that demonstrate the full pipeline without any API costs.
"""

from __future__ import annotations

import json
import re
import math
import config

# ── Mode detection ──────────────────────────────────────────────────────────

USE_MOCK = not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY.startswith("your-")

if not USE_MOCK:
    import anthropic
    _client: anthropic.Anthropic | None = None

    def get_client() -> anthropic.Anthropic:
        global _client
        if _client is None:
            _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return _client
else:
    print("[LLM Service] No Anthropic API key — using mock mode (heuristic scoring)")


# ── Topic Extraction ────────────────────────────────────────────────────────

def extract_topics(title: str, abstract: str, keywords: list[str]) -> dict:
    """Extract structured research topics from a paper."""
    if USE_MOCK:
        return _mock_extract_topics(title, abstract, keywords)
    return _live_extract_topics(title, abstract, keywords)


def _mock_extract_topics(title: str, abstract: str, keywords: list[str]) -> dict:
    """
    Heuristic topic extraction using keyword analysis and n-gram patterns.
    No LLM needed — parses the text directly.
    """
    text = f"{title} {abstract}".lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    word_freq = {}
    for w in words:
        if w not in _STOPWORDS:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Extract bigrams for multi-word topics
    bigrams = []
    word_list = re.findall(r'\b[a-z]{3,}\b', text)
    for i in range(len(word_list) - 1):
        if word_list[i] not in _STOPWORDS and word_list[i+1] not in _STOPWORDS:
            bigrams.append(f"{word_list[i]} {word_list[i+1]}")

    bigram_freq = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    # Top single-word terms
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Map to known academic domains
    domains = _match_domains(text)
    methodologies = _match_methodologies(text)

    # Sub-topics from top bigrams + keywords
    sub_topics = [bg for bg, _ in top_bigrams[:5]]
    if keywords:
        sub_topics = [k.lower().strip() for k in keywords[:5]] + sub_topics
        sub_topics = list(dict.fromkeys(sub_topics))[:5]  # Deduplicate

    # Expanded terms from frequent words
    expanded = [w for w, _ in top_words[:8] if w not in [d.split()[-1] for d in domains]]

    return {
        "primary_domains": domains[:4] if domains else ["general science"],
        "methodologies": methodologies[:3] if methodologies else ["empirical study"],
        "sub_topics": sub_topics[:5],
        "expanded_terms": expanded[:8],
        "interdisciplinary_bridges": _detect_bridges(domains),
    }


def _live_extract_topics(title: str, abstract: str, keywords: list[str]) -> dict:
    """Use Claude API for topic extraction."""
    client = get_client()

    prompt = f"""Analyze this academic paper and extract structured information for finding peer reviewers.

Title: {title}
Abstract: {abstract}
Keywords: {', '.join(keywords) if keywords else 'None provided'}

Return a JSON object with these fields:
- "primary_domains": list of 2-4 primary research domains
- "methodologies": list of 1-3 methodologies used
- "sub_topics": list of 3-5 specific sub-topics
- "expanded_terms": list of 5-8 related search terms a reviewer might publish about
- "interdisciplinary_bridges": list of 0-2 fields this paper bridges

Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    return json.loads(text)


# ── Re-ranking ──────────────────────────────────────────────────────────────

def rerank_candidates(
    title: str,
    abstract: str,
    keywords: list[str],
    candidates: list[dict],
) -> list[dict]:
    """Re-rank and score candidate reviewers."""
    if USE_MOCK:
        return _mock_rerank(title, abstract, keywords, candidates)
    return _live_rerank(title, abstract, keywords, candidates)


def _mock_rerank(
    title: str,
    abstract: str,
    keywords: list[str],
    candidates: list[dict],
) -> list[dict]:
    """
    Heuristic re-ranking based on:
    - Vector similarity score (from Qdrant / numpy cosine)
    - Topic keyword overlap (bidirectional — scored against candidate terms)
    - Research summary text similarity
    - H-index / citation-based seniority
    - Recency of last publication
    """
    # Build query terms from title + abstract + keywords
    query_terms = set()
    for text in [title.lower(), abstract.lower()] + [k.lower() for k in keywords]:
        query_terms.update(re.findall(r'\b[a-z]{3,}\b', text))
    query_terms -= _STOPWORDS

    # Also build query bigrams for phrase-level matching
    query_text = f"{title} {abstract} {' '.join(keywords)}".lower()
    query_words = re.findall(r'\b[a-z]{3,}\b', query_text)
    query_bigrams = set()
    for i in range(len(query_words) - 1):
        if query_words[i] not in _STOPWORDS and query_words[i+1] not in _STOPWORDS:
            query_bigrams.add(f"{query_words[i]} {query_words[i+1]}")

    scored = []
    for candidate in candidates:
        c = candidate.copy()

        # ── Topic score: bidirectional keyword overlap ──
        # Use candidate topic terms as the denominator (not the huge query set)
        candidate_terms = set()
        for topic in c.get("topics", []):
            candidate_terms.update(re.findall(r'\b[a-z]{3,}\b', topic.lower()))
        candidate_terms -= _STOPWORDS

        # Also check research_summary if available
        summary = c.get("research_summary", "")
        if summary:
            summary_terms = set(re.findall(r'\b[a-z]{3,}\b', summary.lower())) - _STOPWORDS
            candidate_terms |= summary_terms

        overlap = len(query_terms & candidate_terms)
        # Score against candidate terms (what % of candidate's expertise matches the query)
        cand_coverage = overlap / max(len(candidate_terms), 1)
        # Also score against a focused subset of query terms (keywords only)
        kw_terms = set()
        for k in keywords:
            kw_terms.update(re.findall(r'\b[a-z]{3,}\b', k.lower()))
        kw_terms -= _STOPWORDS
        kw_overlap = len(kw_terms & candidate_terms) / max(len(kw_terms), 1) if kw_terms else 0

        # Bigram matching for phrase-level accuracy
        cand_text = " ".join(c.get("topics", [])).lower() + " " + summary.lower()
        cand_bigrams = set()
        cw = re.findall(r'\b[a-z]{3,}\b', cand_text)
        for i in range(len(cw) - 1):
            if cw[i] not in _STOPWORDS and cw[i+1] not in _STOPWORDS:
                cand_bigrams.add(f"{cw[i]} {cw[i+1]}")
        bigram_overlap = len(query_bigrams & cand_bigrams) / max(len(cand_bigrams), 1) if cand_bigrams else 0

        # Phrase-in-topic matching: check if user keyword words appear in candidate topic names
        # Handles cases like keyword "seismic inversion" matching topic "Seismic Imaging and Inversion Techniques"
        topic_phrases = [t.lower() for t in c.get("topics", [])]
        phrase_hits = 0
        for kw in keywords:
            kw_words = set(re.findall(r'\b[a-z]{3,}\b', kw.lower())) - _STOPWORDS
            if not kw_words:
                continue
            for tp in topic_phrases:
                tp_words = set(re.findall(r'\b[a-z]{3,}\b', tp)) - _STOPWORDS
                # If most keyword words appear in this topic, count it as a hit
                if len(kw_words & tp_words) >= max(len(kw_words) * 0.5, 1):
                    phrase_hits += 1
                    break
        phrase_match = phrase_hits / max(len(keywords), 1) if keywords else 0

        # Combined topic score: blend coverage metrics
        raw_topic = (
            cand_coverage * 0.20
            + kw_overlap * 0.25
            + bigram_overlap * 0.20
            + phrase_match * 0.35
        ) * 10
        # Boost by vector similarity (semantic signal — captures meaning beyond keywords)
        vector_sim = c.get("score", 0)
        # Use vector similarity as a floor — if semantic match is strong, topic can't be too low
        vec_topic_floor = max((vector_sim - 0.25) / 0.45, 0) * 10  # 0.25→0, 0.70→10
        topic_score = min(max(raw_topic * 0.50 + vector_sim * 10 * 0.50, vec_topic_floor), 10)

        # ── Methodology score: vector similarity is our best semantic proxy ──
        # Scale cosine similarity (typical range 0.3–0.75) to 0–10
        methodology_score = min(max((vector_sim - 0.2) / 0.5, 0) * 10, 10)

        # ── Seniority score: from h-index (smoothed curve) ──
        h = c.get("h_index", 0) or 0
        if h >= 50:
            seniority_score = 9.8
        elif h >= 40:
            seniority_score = 9.5
        elif h >= 30:
            seniority_score = 9.0
        elif h >= 25:
            seniority_score = 8.5
        elif h >= 18:
            seniority_score = 8.0
        elif h >= 12:
            seniority_score = 7.5
        elif h >= 8:
            seniority_score = 7.0
        elif h >= 5:
            seniority_score = 6.0
        elif h >= 3:
            seniority_score = 5.0
        else:
            seniority_score = 3.5

        # ── Recency score: based on last publication date ──
        last_pub = c.get("last_publication_date", "")
        if last_pub and len(last_pub) >= 4:
            try:
                pub_year = int(last_pub[:4])
                years_ago = 2026 - pub_year
                if years_ago <= 0:
                    recency_score = 9.8
                elif years_ago <= 1:
                    recency_score = 9.5
                elif years_ago <= 2:
                    recency_score = 8.5
                elif years_ago <= 3:
                    recency_score = 7.5
                elif years_ago <= 5:
                    recency_score = 5.5
                else:
                    recency_score = 3.0
            except ValueError:
                recency_score = 5.0
        else:
            recency_score = 5.0

        # ── Weighted overall score ──
        overall = (
            topic_score * 0.35
            + methodology_score * 0.30
            + seniority_score * 0.15
            + recency_score * 0.20
        )

        # ── Generate reasoning ──
        reasoning_parts = []
        if topic_score >= 7:
            reasoning_parts.append(f"Strong topic alignment ({overlap} matching terms)")
        elif topic_score >= 4:
            reasoning_parts.append(f"Good topic relevance ({overlap} matching terms)")
        elif overlap > 0:
            reasoning_parts.append(f"Some topic overlap ({overlap} matching terms)")
        else:
            reasoning_parts.append("Related domain expertise")

        if methodology_score >= 7:
            reasoning_parts.append("strong methodological match")
        elif methodology_score >= 4:
            reasoning_parts.append("relevant methodological expertise")

        if h >= 25:
            reasoning_parts.append(f"senior researcher (h-index: {h})")
        elif h >= 12:
            reasoning_parts.append(f"established researcher (h-index: {h})")
        elif h >= 5:
            reasoning_parts.append(f"active researcher (h-index: {h})")

        if last_pub and last_pub >= "2024":
            reasoning_parts.append("actively publishing")

        c["topic_score"] = round(topic_score, 1)
        c["methodology_score"] = round(methodology_score, 1)
        c["seniority_score"] = round(seniority_score, 1)
        c["recency_score"] = round(recency_score, 1)
        c["overall_score"] = round(overall, 1)
        c["reasoning"] = ". ".join(reasoning_parts).capitalize() + "."
        scored.append(c)

    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    return scored


def _live_rerank(
    title: str,
    abstract: str,
    keywords: list[str],
    candidates: list[dict],
) -> list[dict]:
    """Use Claude API for re-ranking."""
    client = get_client()

    candidate_summaries = []
    for i, c in enumerate(candidates):
        summary = (
            f"Candidate {i+1}:\n"
            f"  Name: {c['name']}\n"
            f"  Institution: {c.get('institution', 'Unknown')}\n"
            f"  Topics: {', '.join(c.get('topics', [])[:8])}\n"
            f"  H-index: {c.get('h_index', 'N/A')}\n"
            f"  Citations: {c.get('citation_count', 'N/A')}\n"
            f"  Works: {c.get('works_count', 'N/A')}\n"
            f"  Last publication: {c.get('last_publication_date', 'N/A')}\n"
            f"  Vector similarity: {c.get('score', 0):.3f}"
        )
        candidate_summaries.append(summary)

    candidates_text = "\n\n".join(candidate_summaries)

    prompt = f"""You are an expert academic editor finding peer reviewers for a paper.

PAPER:
Title: {title}
Abstract: {abstract}
Keywords: {', '.join(keywords) if keywords else 'None'}

CANDIDATE REVIEWERS (from semantic search):
{candidates_text}

For each candidate, score them on these dimensions (0-10 scale):
- topic_score: How well their research topics align with this paper
- methodology_score: Whether they have expertise in the methods used
- seniority_score: Whether their h-index/citations suggest appropriate seniority to review
- recency_score: Whether they've published recently in this area

Then compute overall_score as a weighted average: topic(0.4) + methodology(0.25) + seniority(0.15) + recency(0.2)

Also provide a 1-2 sentence "reasoning" explaining why they would or wouldn't be a good reviewer.

Return a JSON array sorted by overall_score descending. Each element:
{{
  "candidate_index": <int, 0-based>,
  "topic_score": <float>,
  "methodology_score": <float>,
  "seniority_score": <float>,
  "recency_score": <float>,
  "overall_score": <float>,
  "reasoning": "<string>"
}}

Return ONLY the JSON array, no other text. Include ALL candidates."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    rankings = json.loads(text)

    scored = []
    for rank in rankings:
        idx = rank["candidate_index"]
        if 0 <= idx < len(candidates):
            candidate = candidates[idx].copy()
            candidate["topic_score"] = rank.get("topic_score", 0)
            candidate["methodology_score"] = rank.get("methodology_score", 0)
            candidate["seniority_score"] = rank.get("seniority_score", 0)
            candidate["recency_score"] = rank.get("recency_score", 0)
            candidate["overall_score"] = rank.get("overall_score", 0)
            candidate["reasoning"] = rank.get("reasoning", "")
            scored.append(candidate)

    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    return scored


# ── Heuristic helpers ───────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "has", "have", "been", "from", "this", "that",
    "with", "they", "will", "each", "make", "like", "into", "over", "such",
    "than", "them", "then", "these", "some", "would", "other", "about", "which",
    "their", "there", "could", "more", "also", "most", "here", "both", "after",
    "those", "using", "used", "based", "show", "shown", "well", "however",
    "between", "through", "where", "while", "during", "before", "should",
    "results", "paper", "study", "method", "methods", "approach", "propose",
    "proposed", "present", "presented", "demonstrate", "existing", "recent",
    "first", "second", "new", "novel", "different", "important", "significant",
    "provide", "provides", "including", "across", "within", "without",
    "performance", "compared", "model", "models", "data", "analysis",
}

_DOMAIN_PATTERNS = {
    "machine learning": ["machine learning", "deep learning", "neural network", "supervised", "unsupervised", "reinforcement learning", "classification", "regression"],
    "natural language processing": ["natural language", "nlp", "text mining", "language model", "sentiment", "named entity", "parsing", "translation", "tokeniz"],
    "computer vision": ["computer vision", "image recognition", "object detection", "segmentation", "convolutional", "visual", "image classification"],
    "genomics": ["genome", "genomic", "dna", "rna", "sequencing", "gene expression", "transcriptom", "epigenom"],
    "neuroscience": ["neuroscience", "neural", "brain", "cognitive", "fmri", "eeg", "neuroimaging", "synaptic"],
    "climate science": ["climate", "global warming", "greenhouse", "carbon", "atmospheric", "temperature anomal"],
    "public health": ["epidemiol", "public health", "pandemic", "vaccine", "mortality", "morbidity", "disease surveillance"],
    "materials science": ["materials science", "nanostructur", "polymer", "alloy", "crystallin", "thin film"],
    "quantum computing": ["quantum comput", "qubit", "quantum circuit", "quantum entangle", "superposition"],
    "astrophysics": ["astrophysic", "stellar", "galaxy", "cosmolog", "exoplanet", "dark matter", "gravitational"],
    "renewable energy": ["solar cell", "wind energy", "renewable", "photovoltaic", "energy storage", "battery"],
    "economics": ["economic", "market", "inflation", "monetary", "fiscal", "behavioral economics"],
    "chemistry": ["chemical", "molecular", "synthesis", "catalyst", "organic chemistry", "reaction mechanism"],
    "robotics": ["robot", "autonomous", "manipulation", "motion planning", "swarm", "human-robot"],
    "cybersecurity": ["security", "cryptograph", "malware", "intrusion detection", "vulnerability", "encryption"],
    "bioinformatics": ["bioinformatic", "protein structure", "sequence alignment", "phylogenet", "protein folding"],
    "statistics": ["statistical", "bayesian", "regression", "hypothesis test", "probability", "stochastic"],
    "medicine": ["clinical", "patient", "treatment", "diagnosis", "therapeutic", "randomized trial", "placebo"],
}

_METHOD_PATTERNS = {
    "deep learning": ["deep learning", "neural network", "cnn", "rnn", "lstm", "transformer", "attention mechanism", "backpropagation"],
    "statistical analysis": ["statistical", "regression", "anova", "chi-square", "t-test", "confidence interval", "p-value"],
    "randomized controlled trial": ["randomized", "controlled trial", "rct", "placebo", "double-blind"],
    "survey methodology": ["survey", "questionnaire", "likert", "respondent"],
    "simulation": ["simulation", "monte carlo", "agent-based", "finite element"],
    "qualitative analysis": ["qualitative", "interview", "thematic analysis", "grounded theory"],
    "meta-analysis": ["meta-analysis", "systematic review", "effect size", "heterogeneity"],
    "experimental": ["experiment", "laboratory", "controlled experiment", "in vitro", "in vivo"],
    "computational modeling": ["computational model", "numerical", "differential equation", "optimization"],
    "transfer learning": ["transfer learning", "fine-tun", "pre-train", "domain adaptation"],
}


def _match_domains(text: str) -> list[str]:
    """Match text against known academic domains."""
    scores = {}
    for domain, patterns in _DOMAIN_PATTERNS.items():
        score = sum(1 for p in patterns if p in text)
        if score > 0:
            scores[domain] = score
    return sorted(scores, key=scores.get, reverse=True)


def _match_methodologies(text: str) -> list[str]:
    """Match text against known research methodologies."""
    scores = {}
    for method, patterns in _METHOD_PATTERNS.items():
        score = sum(1 for p in patterns if p in text)
        if score > 0:
            scores[method] = score
    return sorted(scores, key=scores.get, reverse=True)


def _detect_bridges(domains: list[str]) -> list[str]:
    """Detect interdisciplinary bridges from domain combinations."""
    bridges = []
    domain_set = set(domains)

    bridge_map = {
        frozenset(["machine learning", "medicine"]): "medical AI",
        frozenset(["machine learning", "genomics"]): "computational genomics",
        frozenset(["machine learning", "materials science"]): "materials informatics",
        frozenset(["statistics", "genomics"]): "statistical genetics",
        frozenset(["neuroscience", "machine learning"]): "computational neuroscience",
        frozenset(["economics", "machine learning"]): "computational economics",
        frozenset(["chemistry", "machine learning"]): "cheminformatics",
        frozenset(["climate science", "statistics"]): "climate modeling",
        frozenset(["robotics", "machine learning"]): "intelligent robotics",
    }

    for combo, bridge in bridge_map.items():
        if combo.issubset(domain_set):
            bridges.append(bridge)

    return bridges[:2]
