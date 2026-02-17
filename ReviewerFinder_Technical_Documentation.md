# ReviewerFinder AI - Technical Documentation

## System Overview

ReviewerFinder AI is an AI-native peer reviewer discovery system that helps journal editors and researchers find the most suitable peer reviewers for academic manuscripts. Given a paper's title, abstract, and keywords, the system uses vector similarity search over a pre-indexed database of author profiles, followed by multi-dimensional scoring and ranking, to produce a ranked list of expert reviewers with contact information and conflict-of-interest flags.

---

## Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit (Python) | Interactive web UI with 3-stage workflow |
| Vector Database | Qdrant (Docker, v1.7.4) | Approximate Nearest Neighbor search over author embeddings |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dimensional dense vector representations |
| Scoring Engine | Heuristic-based (mock LLM) | Multi-signal reviewer scoring |
| Data Source | OpenAlex API | Author profiles, publications, affiliations |
| Contact Enrichment | ORCID API + OpenAlex API | Email, homepage, profile links |
| Data Storage | JSON file (`data/authors.json`) | Prototype author database |

### System Flow

```
Manuscript Upload
       |
       v
Text Extraction (PDF/DOCX/TXT)
       |
       v
Title + Abstract + Keywords Parsing
       |
       v
Topic Extraction (LLM/Heuristic)
       |
       v
Query Embedding Generation (384-dim vector)
       |
       v
Vector Similarity Search (Qdrant ANN)
       |
       v
Candidate Scoring & Re-ranking
       |
       v
Contact Enrichment (Email, ORCID, OpenAlex)
       |
       v
Conflict of Interest Detection
       |
       v
Ranked Reviewer List with Scores, Contacts, COI Flags
```

---

## Data Pipeline: How the Author Database is Built

The system operates on a pre-indexed database of academic author profiles sourced from OpenAlex, an open-access scholarly metadata catalog.

### Step 1: Author Ingestion from OpenAlex

The pipeline queries the OpenAlex API across 15 diverse research domains to build a representative author pool:

**Seed Topics:**
- Machine Learning & Deep Learning
- Natural Language Processing
- Computer Vision
- Climate Science & Atmospheric Research
- Genomics & Bioinformatics
- Quantum Computing
- Neuroscience & Brain Imaging
- Renewable Energy & Solar Cells
- Public Health & Epidemiology
- Materials Science & Nanotechnology
- Astrophysics & Cosmology
- Economics & Econometrics
- Chemistry & Catalysis
- Robotics & Control Systems
- Cybersecurity & Cryptography

For each topic, the system fetches approximately 19 authors (282 total in the prototype), retrieving:
- Author name, unique OpenAlex ID
- ORCID identifier (if available)
- Institutional affiliations (name, country, ROR ID)
- Research topics (up to 15 per author)
- Publication metrics: h-index, citation count, works count
- Last publication date

### Step 2: Research Summary Construction

For each author, the system fetches up to 5 recent publications with abstracts from OpenAlex. Abstracts are reconstructed from OpenAlex's inverted index format (a word-to-positions dictionary) back into readable text. These abstracts are concatenated to form a `research_summary` field (capped at 5,000 characters) that represents the author's recent research focus.

**Why this matters:** The research summary is the primary text used to generate the author's embedding vector. Richer summaries produce more accurate semantic matching.

### Step 3: Embedding Generation

Each author's profile is converted into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` sentence-transformer model.

**Input text formula:**
```
{institution} | {topic1}, {topic2}, ... | {research_summary[:2000]}
```

This combines institutional context, declared research topics, and actual publication content into a single semantic representation.

**Model details:**
- Model: `all-MiniLM-L6-v2` (prototype) / `allenai/specter2` (production)
- Dimensions: 384
- Normalization: Vectors are L2-normalized for cosine similarity
- Batch size: 32 authors per batch

### Step 4: Vector Indexing in Qdrant

The generated embeddings are uploaded to a Qdrant collection (`author_embeddings`) with:
- Distance metric: Cosine similarity
- Vector size: 384 dimensions
- Payload: author metadata (name, institution, topics, h-index, etc.)
- Point ID: Deterministic UUID5 hash of the OpenAlex author ID

---

## Search Pipeline: How Reviewers are Found

When a user uploads a manuscript and initiates a search, the system executes a 6-step pipeline:

### Step 1: Topic Extraction

The system analyzes the manuscript's title, abstract, and keywords to extract structured research topics:

| Field | Description | Example |
|-------|-------------|---------|
| `primary_domains` | Main research areas | "Machine Learning", "Genomics" |
| `methodologies` | Research methods used | "Deep Learning", "Statistical Analysis" |
| `sub_topics` | Specific sub-areas | "Transformer Architecture", "Gene Expression" |
| `expanded_terms` | Broader related terms | "Artificial Intelligence", "Bioinformatics" |
| `interdisciplinary_bridges` | Cross-domain connections | "Computational Biology" |

In mock mode (no API key), topics are extracted via keyword analysis from the title and abstract.

### Step 2: Query Embedding

The manuscript text (title + abstract + keywords) is converted into a 384-dimensional vector using the same `all-MiniLM-L6-v2` model used during indexing. This ensures the query vector lives in the same semantic space as the author vectors.

### Step 3: Vector Similarity Search (Qdrant)

The query vector is sent to Qdrant for Approximate Nearest Neighbor (ANN) search.

**This is where "Search Depth (Candidates)" comes in:**

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| **Search Depth (Candidates)** | 50 | 20-100 | How many candidate authors to retrieve from the vector database before scoring |

A higher search depth casts a wider net, finding more potential matches but taking slightly longer. A search depth of 50 means Qdrant returns the 50 authors whose research embedding vectors are most similar to the manuscript's embedding vector.

**Filter applied:** Only authors with 3 or more published works are considered (filters out inactive or junior researchers).

### Step 4: Candidate Scoring & Re-ranking

The top candidates (capped at 30 for scoring context) are scored across 4 dimensions:

#### Scoring Dimensions

**1. Topic Score (Weight: 40%)**
- Measures overlap between the manuscript's keywords/topics and the candidate's declared research topics
- Calculation: Extracts meaningful words (4+ characters, excluding common stopwords) from the manuscript title and abstract, then counts how many match the candidate's topics
- Formula: `(matching_terms / total_query_terms) * 15`, capped at 10.0
- High score = the candidate publishes in the same research area as the manuscript

**2. Methodology Score (Weight: 25%)**
- Approximated from the vector similarity score returned by Qdrant
- Formula: `vector_similarity * 12`, capped at 10.0
- High score = the candidate's research content is semantically close to the manuscript

**3. Seniority Score (Weight: 15%)**
- Based on the candidate's h-index:

| H-index | Score |
|---------|-------|
| >= 40 | 9.5 |
| >= 25 | 8.5 |
| >= 15 | 7.5 |
| >= 8 | 6.5 |
| >= 3 | 5.0 |
| < 3 | 3.0 |

- High score = established researcher with significant publication impact

**4. Recency Score (Weight: 20%)**
- Based on years since the candidate's last publication:

| Years Since Last Pub | Score |
|---------------------|-------|
| <= 1 year | 9.5 |
| <= 2 years | 8.5 |
| <= 3 years | 7.0 |
| <= 5 years | 5.0 |
| > 5 years | 3.0 |

- High score = the candidate is currently active in research

#### Overall Score Calculation

```
overall_score = (topic_score * 0.40)
              + (methodology_score * 0.25)
              + (seniority_score * 0.15)
              + (recency_score * 0.20)
```

The overall score ranges from 0.0 to 10.0.

**This is where "Number of Reviewers" comes in:**

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| **Number of Reviewers** | 10 | 3-30 | How many top-ranked reviewers to display in the final results |

After scoring all candidates, the system sorts by overall_score descending and returns the top N reviewers.

### Step 5: Contact Enrichment

Each selected reviewer's contact information is enriched through a 3-tier lookup:

#### Tier 1: Local Database (`data/authors.json`)
- Checks the pre-seeded author profiles for stored contact info
- May contain: email, ORCID ID, homepage, Google Scholar link
- Also extracts co-author IDs and affiliations for COI detection

#### Tier 2: OpenAlex Author API
- If email is still missing, queries `https://api.openalex.org/authors/{id}`
- Retrieves: ORCID ID (if not already known), institutional ROR page
- Generates: OpenAlex profile URL (`https://openalex.org/{author_id}`)

#### Tier 3: ORCID Public API
- If an ORCID ID exists but email is still missing, queries `https://pub.orcid.org/v3.0/{orcid_id}/person`
- Retrieves: public email address (if the researcher has made it public), researcher URLs
- Extracts homepage, Google Scholar link from researcher-urls section

#### What If No Email is Found?

Many researchers do not make their email publicly available through OpenAlex or ORCID. In such cases:

1. The reviewer card displays **available profile links** (ORCID, OpenAlex, Homepage) so the editor can find contact information by visiting those profiles
2. The **Invite button is hidden** for reviewers without email addresses
3. The **export (CSV/Excel)** includes all available contact fields, with the Email column left blank

**In the prototype**, all 282 seeded authors have been enriched with generated email addresses (derived from first initial + last name @ institution domain) to demonstrate the full invite workflow. In a production system, emails would come from:
- ORCID public profiles (if researcher opted in)
- Institutional directories (via web scraping or partnerships)
- Publisher databases (via API integrations)

### Step 6: Conflict of Interest Detection

Each candidate is checked for potential conflicts of interest with the manuscript authors:

#### COI Type 1: Co-authorship (Severity: HIGH)
- Checks if the candidate has co-authored papers with any of the manuscript's authors
- Data source: `co_author_ids` field from OpenAlex (up to 50 frequent co-authors per person)
- Flag text: *"Has co-authored with paper author (ID: ...)"*

#### COI Type 2: Same Institution (Severity: MEDIUM)
- Checks if the candidate's current institution matches any institution listed by the manuscript authors
- Uses substring matching after normalizing to lowercase
- Flag text: *"Same institution: MIT"*

#### COI Type 3: Name Similarity (Severity: CRITICAL or LOW)
- **Exact name match** (severity: CRITICAL): The candidate's full name matches a manuscript author exactly
- **Last name match** (severity: LOW): The candidate shares a last name (3+ characters) with a manuscript author
- Flag text: *"Candidate name matches paper author: Dr. Jane Smith"*

COI flags are displayed as colored warning badges on each reviewer card:
- RED badge: High/Critical severity
- YELLOW badge: Medium/Low severity

---

## Profile Links Explained

Each reviewer card may display up to three external profile links:

### ORCID
- **What it is:** Open Researcher and Contributor ID - a unique persistent identifier for researchers
- **URL format:** `https://orcid.org/0000-0001-2345-6789`
- **What you'll find:** The researcher's verified publication list, institutional affiliations, and (if public) email address
- **Source:** OpenAlex author metadata or ORCID API lookup

### OpenAlex
- **What it is:** An open-access catalog of scholarly papers, authors, and institutions
- **URL format:** `https://openalex.org/A1234567890`
- **What you'll find:** The author's complete publication history, citation metrics, co-authors, institutional history, and research topics
- **Source:** Generated from the author's OpenAlex ID

### Homepage
- **What it is:** The researcher's personal or institutional webpage
- **URL format:** Varies (e.g., `https://www.mit.edu/~jsmith`)
- **What you'll find:** Lab information, current projects, CV, contact details
- **Source:** ORCID researcher-urls section (looks for URLs labeled "homepage" or "personal")

---

## User Interface: 3-Stage Workflow

### Stage 1: Upload
- User uploads a PDF, DOCX, or TXT manuscript file
- The system automatically extracts the title, abstract, and keywords using text parsing

### Stage 2: Review & Configure
- User reviews and edits the extracted title, abstract, and keywords
- Configures search parameters:
  - **Number of reviewers** (3-30): Final reviewer count
  - **Search depth** (20-100): Vector search breadth
- Optional COI filters: Paper author names and institutions

### Stage 3: Results
- Displays ranked reviewer cards with:
  - Overall score badge (color-coded: green >= 7, yellow >= 5, red < 5)
  - Four dimension score bars (Topic, Method, Seniority, Recency)
  - Author metadata (institution, h-index, citations, works count)
  - Research topic tags
  - AI reasoning text explaining why this reviewer was selected
  - Contact information (email + profile links)
  - COI warning flags (if any)
  - Invite button (opens pre-filled email template)
- Export options: CSV and Excel download

---

## Invite Workflow

For reviewers with available email addresses, the system provides a one-click invite feature:

1. Click the **Invite** button on any reviewer card
2. An inline compose panel opens with:
   - **To:** Pre-filled with the reviewer's email
   - **Subject:** Pre-filled as *"Invitation to Review: {paper title}"*
   - **Body:** Pre-filled professional template mentioning:
     - The manuscript title
     - The reviewer's specific expertise areas (from their topic tags)
     - A request to review
3. User can edit the subject and body
4. Click **Send** to mark the invite as sent (visual confirmation badge)

*Note: In the current prototype, the Send action is simulated (marks as sent in the UI session state). Production integration would connect to an email service (SMTP, SendGrid, etc.).*

---

## Configuration Parameters Summary

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Number of reviewers | UI slider | 10 | Final count of ranked reviewers returned |
| Search depth (candidates) | UI slider | 50 | Number of vector search candidates before scoring |
| Paper author names | COI filter | (empty) | Comma-separated names for conflict detection |
| Author institutions | COI filter | (empty) | Comma-separated institutions for conflict detection |
| Qdrant host | Environment | localhost:6333 | Vector database connection |
| Embedding model | embedding_service.py | all-MiniLM-L6-v2 | Sentence transformer model |
| Vector dimensions | Qdrant collection | 384 | Embedding vector size |
| Min works filter | search_service.py | 3 | Minimum publications to be considered |
| Max LLM candidates | search_service.py | 30 | Cap on candidates sent to scoring |

---

## Production Considerations

The current prototype is designed for demonstration with 282 authors. For production scale:

| Aspect | Prototype | Production |
|--------|-----------|------------|
| Author count | 282 | 10K - 1M+ |
| Database | JSON file | PostgreSQL |
| Vector DB | Qdrant (single node) | Qdrant cluster / Pinecone |
| Embedding model | all-MiniLM-L6-v2 (384d) | allenai/specter2 (768d) |
| Scoring | Heuristic mock | Claude API (real LLM re-ranking) |
| Email service | Simulated | SMTP / SendGrid integration |
| Authentication | None | OAuth / API keys |
| Contact enrichment | Batch pre-computed | Real-time API calls with caching |
| Task processing | Synchronous | Celery async workers |

---

## File Structure

```
RFS/
+-- app.py                          # Streamlit UI (3-stage workflow)
+-- .streamlit/config.toml          # Streamlit dark theme configuration
+-- data/
|   +-- authors.json                # 282 seeded author profiles
+-- services/
|   +-- search_service.py           # Main search pipeline orchestrator
|   +-- embedding_service.py        # Query embedding generation
|   +-- llm_service.py              # Topic extraction + scoring (mock mode)
|   +-- contact_service.py          # Contact enrichment (local DB + ORCID + OpenAlex)
|   +-- coi_service.py              # Conflict of interest detection
+-- pipeline/
|   +-- seed_prototype.py           # All-in-one seeding script
|   +-- ingest_openalex.py          # Fetch authors from OpenAlex API
|   +-- build_embeddings.py         # Generate sentence-transformer embeddings
|   +-- index_qdrant.py             # Upload vectors to Qdrant
```
