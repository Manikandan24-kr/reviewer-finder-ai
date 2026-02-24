"""
AI Reviewer Finder — Dark Theme UI
Dark glassmorphism + neon accents + premium feel.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import os, sys, time, re, html as html_mod
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ReviewerFinder AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── helpers (BEFORE main flow) ────────────────────────────────────────────

def extract_text_from_pdf(file):
    from PyPDF2 import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages[:15])

def extract_text_from_docx(file):
    from docx import Document
    return "\n".join(p.text for p in Document(file).paragraphs)

def parse_manuscript(text: str) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = abstract = ""
    keywords: list[str] = []
    authors: list[str] = []
    author_institutions: list[str] = []
    for line in lines[:10]:
        if len(line) > 15 and not line.lower().startswith(("abstract","keyword","introduction","http","doi")):
            title = line; break
    full = text.lower()
    # Extract authors: lines between title and abstract that contain names/emails
    title_found = False
    for line in lines[:20]:
        if line == title:
            title_found = True; continue
        if not title_found:
            continue
        ll = line.lower()
        if ll.startswith(("abstract","a b s t r a c t")):
            break
        # Skip lines that are just "Corresponding author:" references
        if re.match(r'^corresponding\s*(author)?', ll):
            continue
        # Look for author-like lines (contain names, possibly with affiliations)
        # Extract name part before comma/institution info
        if len(line) > 3 and len(line) < 300 and not ll.startswith(("keywords","key words","e-mail","email","doi","http","received","accepted","published")):
            # Try to extract just the name portion
            name_part = re.split(r'[,;]', line)[0].strip()
            # Remove email addresses
            name_part = re.sub(r'\S+@\S+', '', name_part).strip()
            # Remove E-mail: prefix and trailing text
            name_part = re.sub(r'E-mail:.*', '', name_part, flags=re.IGNORECASE).strip()
            # Remove common suffixes
            name_part = re.sub(r'\b(Ph\.?D|M\.?D|Dr\.?|Prof\.?)\b', '', name_part, flags=re.IGNORECASE).strip()
            name_part = name_part.strip('., ')
            if name_part and len(name_part) > 3 and len(name_part) < 80 and ' ' in name_part:
                # Check it looks like a name (mostly letters)
                alpha_ratio = sum(c.isalpha() or c == ' ' for c in name_part) / max(len(name_part), 1)
                if alpha_ratio > 0.8:
                    authors.append(name_part)
            # Try to extract institution/affiliation — the part right after the name
            parts = re.split(r'[,;]', line)
            for pi, raw_part in enumerate(parts[1:]):
                part = raw_part.strip()
                # Remove emails, addresses (numbers), "E-mail:" text
                part = re.sub(r'\S+@\S+', '', part).strip()
                part = re.sub(r'E-mail:.*', '', part, flags=re.IGNORECASE).strip()
                part = part.strip('., ')
                if not part or len(part) < 4:
                    continue
                # Check for institution keywords OR treat 2nd comma-part as org name
                inst_keywords = ["university","institute","college","laboratory","lab","department",
                                 "dept","school","centre","center","hospital","corporation","corp",
                                 "survey","agency","research","national","organisation","organization"]
                if any(ik in part.lower() for ik in inst_keywords) and len(part) > 5:
                    author_institutions.append(part)
                    break
                # If it's the first part after the name and looks like an org (has uppercase, > 4 chars)
                elif pi == 0 and len(part) > 5 and part[0].isupper() and not any(c.isdigit() for c in part[:5]):
                    author_institutions.append(part)
                    break
    for marker in ["abstract","a b s t r a c t"]:
        idx = full.find(marker)
        if idx != -1:
            at = text[idx+len(marker):idx+len(marker)+3000].strip()
            at = re.sub(r'^[\s:\-—\.]+','',at)
            for em in ["keywords","key words","introduction","1.","1 ","index terms"]:
                ei = at.lower().find(em)
                if ei > 50: at = at[:ei].strip(); break
            abstract = at[:2000]; break
    for marker in ["keywords","key words","index terms"]:
        idx = full.find(marker)
        if idx != -1:
            kt = text[idx+len(marker):idx+len(marker)+500].strip()
            kt = re.sub(r'^[\s:\-—\.]+','',kt)
            for em in ["\n\n","introduction","1.","1 "]:
                ei = kt.lower().find(em)
                if ei > 5: kt = kt[:ei]; break
            keywords = [k.strip().rstrip('.') for k in re.split(r'[;,\n]',kt) if 2<len(k.strip())<60][:10]
            break

    # Auto-generate keywords from title + abstract if none were found
    if not keywords and (title or abstract):
        keywords = _extract_keywords_from_text(title, abstract)

    # Also try to extract institution from author lines with email patterns
    if not author_institutions and authors:
        for line in lines[:20]:
            for inst_kw in ["university","institute","college","laboratory","lab","centre","center","survey","corporation"]:
                if inst_kw in line.lower() and len(line) > 10:
                    # Extract the institution part
                    parts = re.split(r'[,;]', line)
                    for part in parts:
                        part = part.strip()
                        if inst_kw in part.lower() and len(part) > 5:
                            clean = re.sub(r'\S+@\S+', '', part).strip().rstrip('., ')
                            if clean and len(clean) > 5:
                                author_institutions.append(clean)
                    break

    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "authors": authors,
        "author_institutions": list(dict.fromkeys(author_institutions)),  # deduplicate
    }


def _extract_keywords_from_text(title: str, abstract: str) -> list[str]:
    """Extract relevant keywords from title and abstract using NLP heuristics."""
    text = f"{title} {abstract}".lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)

    # Stopwords for academic text
    stops = {
        "the","and","for","are","but","not","you","all","can","had","her","was",
        "one","our","out","has","have","been","from","this","that","with","they",
        "will","each","make","like","into","over","such","than","them","then",
        "these","some","would","other","about","which","their","there","could",
        "more","also","most","here","both","after","those","using","used","based",
        "show","shown","well","however","between","through","where","while",
        "during","before","should","results","paper","study","method","methods",
        "approach","propose","proposed","present","presented","demonstrate",
        "existing","recent","first","second","new","novel","different","important",
        "significant","provide","provides","including","across","within","without",
        "performance","compared","model","models","data","analysis","often",
        "when","does","being","value","values","case","cases","effect","effects",
        "test","tests","suggests","suggesting","particularly","may","terms",
        "strongly","simple","address","begin","begins","combination","attempt",
        "attempts","prior","assumption","derive","derived","improve","improved",
        "practical","application","confirmed","offer","offering","serve",
        "serving","useful","efficient","transparent","alternative",
    }

    # Extract bigrams (two-word phrases)
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stops and words[i+1] not in stops:
            bigrams.append(f"{words[i]} {words[i+1]}")

    # Count frequencies
    bigram_freq = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    word_freq = {}
    for w in words:
        if w not in stops and len(w) > 3:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Top bigrams as keywords
    top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [bg for bg, cnt in top_bigrams if cnt >= 2][:6]

    # Add top single words if we don't have enough
    if len(keywords) < 4:
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for w, _ in top_words:
            if w not in " ".join(keywords) and len(keywords) < 8:
                keywords.append(w)

    return keywords[:8]

def _to_excel(df):
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Reviewers")
    return buf.getvalue()

def esc(s: str) -> str:
    return html_mod.escape(str(s))

def _step_indicator(active: int):
    labels = ["Upload", "Review", "Results"]
    dots = []
    for i, label in enumerate(labels, 1):
        if i < active:
            cls, num = "done", "✓"
        elif i == active:
            cls, num = "active", str(i)
        else:
            cls, num = "pending", str(i)
        dots.append(f'<div class="step {cls}"><div class="step-dot {cls}">{num}</div>{label}</div>')
    steps_html = '<span class="step-arrow">›</span>'.join(dots)
    st.markdown(f"""
    <div class="unified-header">
        <div class="navbar-left">
            <div class="navbar-logo">RF</div>
            <div class="navbar-title">ReviewerFinder</div>
        </div>
        <div class="steps-inline">{steps_html}</div>
        <div class="navbar-badge">AI Powered</div>
    </div>
    """, unsafe_allow_html=True)

def _score_bars(scores: dict):
    """Render all 4 score bars as a single compact HTML snippet."""
    items = [("Topic","topic_score"),("Method","methodology_score"),("Seniority","seniority_score"),("Recency","recency_score")]
    html = ""
    for label, key in items:
        v = scores.get(key, 0)
        pct = min(v * 10, 100)
        if v >= 7:
            grad = "linear-gradient(90deg, #10b981, #34d399)"
            tc = "#34d399"
        elif v >= 5:
            grad = "linear-gradient(90deg, #f59e0b, #fbbf24)"
            tc = "#fbbf24"
        else:
            grad = "linear-gradient(90deg, #ef4444, #f87171)"
            tc = "#f87171"
        html += f"""<div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
                <span style="font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">{label}</span>
                <span style="font-size:12px;font-weight:800;color:{tc};">{v:.1f}</span>
            </div>
            <div style="height:7px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{grad};border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1);"></div>
            </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────

for k, v in {"stage": "upload", "parsed": {}, "results": None, "invite_open": {}, "invite_sent": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── global styles (dark theme) ───────────────────────────────────────────

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
#MainMenu, footer, header {visibility:hidden}
.stDeployButton {display:none}
section[data-testid="stSidebar"] {display:none}
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0b0f1a;
}
.block-container {
    max-width: 1040px !important;
    padding: 1.5rem 2rem 4rem !important;
}

/* ─── Typography ─── */
h1,h2,h3,h4 {
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.025em !important;
}
p,span,label,div {
    font-family: 'Inter', sans-serif;
}

/* ─── Gradient text ─── */
.gradient-text {
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #c084fc 70%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ─── Unified header ─── */
.unified-header {
    display: flex; align-items: center; gap: 16px;
    padding: 12px 22px; margin-bottom: 24px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
}
.navbar-left {
    display: flex; align-items: center; gap: 10px;
}
.navbar-logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 12px; font-weight: 800;
    box-shadow: 0 2px 12px rgba(99,102,241,.4);
    letter-spacing: -0.02em; flex-shrink: 0;
}
.navbar-title {
    font-size: 16px; font-weight: 800; color: #e2e8f0;
    letter-spacing: -0.02em; white-space: nowrap;
}
.steps-inline {
    display: flex; align-items: center; gap: 6px;
    margin-left: auto;
}
.navbar-badge {
    padding: 5px 14px;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 99px; font-size: 11px; font-weight: 700;
    letter-spacing: .03em; text-transform: uppercase;
    white-space: nowrap; flex-shrink: 0; margin-left: 12px;
}
.step {
    display: flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 600;
    transition: all .3s;
}
.step-dot {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    transition: all .4s cubic-bezier(.4,0,.2,1);
}
.step-dot.done { background: linear-gradient(135deg, #10b981, #059669); color: #fff; box-shadow: 0 2px 8px rgba(16,185,129,.35); }
.step-dot.active { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; box-shadow: 0 2px 10px rgba(99,102,241,.4); }
.step-dot.pending { background: rgba(255,255,255,0.06); color: #64748b; }
.step.done { color: #34d399; }
.step.active { color: #a5b4fc; }
.step.pending { color: #475569; }
.step-arrow { color: #334155; font-size: 16px; margin: 0 4px; font-weight: 300; }

/* ─── Glass cards (dark) ─── */
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
    transition: all .3s cubic-bezier(.4,0,.2,1);
}
.glass-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
    transform: translateY(-1px);
    border-color: rgba(99,102,241,.2);
}

/* ─── Feature cards ─── */
.feature-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,.2);
    transition: all .35s cubic-bezier(.4,0,.2,1);
}
.feature-card:hover {
    box-shadow: 0 12px 40px rgba(99,102,241,.15), 0 4px 12px rgba(0,0,0,.3);
    border-color: rgba(99,102,241,.25);
    transform: translateY(-4px);
}
.feature-icon {
    width: 56px; height: 56px; border-radius: 16px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 16px;
}

/* ─── Inputs ─── */
.stTextInput label,.stTextArea label,.stSlider label,.stFileUploader label,.stSelectbox label {
    color:#94a3b8 !important;font-weight:600 !important;font-size:13px !important;font-family:'Inter',sans-serif !important;
}
.stTextInput input,
.stTextArea textarea,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #141825 !important;
    background-color: #141825 !important;
    border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important; font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .2s !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,.15) !important;
    background: #141825 !important;
    background-color: #141825 !important;
}
/* placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}

/* ─── Upload card ─── */
.upload-card {
    position: relative;
    border-radius: 20px;
    padding: 2px;
    margin-bottom: -8px;
    overflow: hidden;
}
.upload-glow {
    position: absolute; inset: 0;
    border-radius: 20px;
    background: conic-gradient(from 180deg, #6366f1, #8b5cf6, #c084fc, #6366f1);
    opacity: 0.25;
    animation: glowSpin 4s linear infinite;
}
@keyframes glowSpin {
    to { transform: rotate(360deg); }
}
.upload-inner {
    position: relative;
    background: #0e1220;
    border-radius: 18px;
    padding: 36px 28px 28px;
    text-align: center;
}
.upload-icon-wrap {
    width: 64px; height: 64px; border-radius: 18px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.15);
    display: inline-flex; align-items: center; justify-content: center;
    margin-bottom: 18px;
}

/* ─── File uploader: invisible overlay on upload card ─── */
[data-testid="stFileUploader"] {
    position: relative !important;
    margin-top: -180px !important;
    height: 180px !important;
    z-index: 20 !important;
    opacity: 0 !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] * {
    cursor: pointer !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important; font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 14px rgba(99,102,241,.35), inset 0 1px 0 rgba(255,255,255,.1) !important;
    transition: all .25s cubic-bezier(.4,0,.2,1) !important;
    letter-spacing: .01em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.45), inset 0 1px 0 rgba(255,255,255,.15) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── Download buttons ─── */
.stDownloadButton > button {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(8px) !important;
    color: #a5b4fc !important;
    border: 1.5px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.2) !important;
    transition: all .2s !important;
}
.stDownloadButton > button:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,.15) !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
.streamlit-expanderContent,
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background: #0e1220 !important;
    border: 1.5px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
}
details summary span,
[data-testid="stExpander"] summary span {
    color: #cbd5e1 !important;
}
/* ─── All generic Streamlit containers dark ─── */
[data-testid="stForm"],
[data-testid="stVerticalBlock"],
.element-container {
    color: #e2e8f0;
}

/* ─── Slider ─── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,.4) !important;
}
/* slider filled track */
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #94a3b8 !important;
}
/* track colors: override the red/default */
.stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255,255,255,0.06) !important;
}
.stSlider [data-baseweb="slider"] > div > div > div:first-child {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}
.stSlider p, .stSlider label {
    color: #94a3b8 !important;
}

/* ─── Progress ─── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #c084fc) !important;
    background-size: 200% 200% !important;
    animation: shimmer 2s ease infinite !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.06) !important;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ─── Spinner / status text ─── */
.stSpinner > div { color: #94a3b8 !important; }
.stAlert { background: rgba(255,255,255,0.04) !important; border-color: rgba(255,255,255,0.08) !important; }

/* ─── Mobile: stack Streamlit columns ─── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}

</style>""", unsafe_allow_html=True)

st.markdown("""<style>
/* ─── Stat cards ─── */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 24px; }
.stat-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 22px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,.2);
    transition: all .3s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.3); }
.stat-value {
    font-size: 30px; font-weight: 900;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}
.stat-label { font-size: 11px; font-weight: 600; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }

/* ─── Tags ─── */
.tag { display: inline-block; padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; margin: 3px 4px 3px 0; transition: all .2s; }
.tag:hover { transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,.2); }
.tag-purple { background: rgba(139,92,246,0.12); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.15); }
.tag-blue { background: rgba(59,130,246,0.12); color: #93c5fd; border: 1px solid rgba(59,130,246,0.15); }
.tag-green { background: rgba(16,185,129,0.12); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.15); }
.tag-sm { padding: 3px 10px; font-size: 11px; border-radius: 6px; }

/* ─── Reviewer card ─── */
.reviewer-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06) 20%, rgba(255,255,255,0.06) 80%, transparent);
    margin: 12px 0 16px;
}

/* ─── Score overall badge ─── */
.score-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 52px; height: 52px; border-radius: 50%;
    font-size: 16px; font-weight: 900;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}

/* ─── Invite ─── */
.invite-header {
    background: rgba(99,102,241,0.08);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px; padding: 14px 18px;
    margin: 10px 0 8px;
}
.sent-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.12);
    color: #34d399; padding: 6px 16px; border-radius: 99px;
    font-size: 13px; font-weight: 700;
    border: 1px solid rgba(16,185,129,0.2);
    box-shadow: 0 2px 8px rgba(16,185,129,.1);
}

/* ─── Section labels ─── */
.section-label {
    font-size: 11px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 14px;
}
.section-heading {
    font-size: 16px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 18px; display: flex; align-items: center; gap: 8px;
}

/* ─── COI ─── */
.coi-flag {
    padding: 8px 14px; border-radius: 10px; font-size: 13px;
    font-weight: 500; margin-top: 8px;
    display: flex; align-items: center; gap: 6px;
}

/* ─── Profile Links ─── */
.profile-link {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    font-size: 11px;
    font-weight: 600;
    color: #a5b4fc;
    text-decoration: none;
    margin-right: 4px;
    transition: all .2s;
}
.profile-link:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.25);
}

/* ─── AI Inferred Badge ─── */
.ai-inferred-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 8px;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
    color: #fbbf24;
    letter-spacing: .03em;
    margin-left: 4px;
}

/* ─── Fade-in animation ─── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .5s ease-out both; }
.fade-up-d1 { animation: fadeUp .5s ease-out .1s both; }
.fade-up-d2 { animation: fadeUp .5s ease-out .2s both; }
.fade-up-d3 { animation: fadeUp .5s ease-out .3s both; }

/* ─── Mobile Responsive ─── */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 3rem !important;
    }
    /* Header: stack logo + steps vertically */
    .unified-header {
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px 14px;
    }
    .steps-inline {
        margin-left: 0;
        order: 3;
        width: 100%;
        justify-content: center;
        padding-top: 6px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .navbar-badge { margin-left: auto; }

    /* Hero text smaller */
    .fade-up h1,
    .fade-up [style*="font-size:42px"] {
        font-size: 28px !important;
    }
    .fade-up p {
        font-size: 14px !important;
    }

    /* Stat grid: 1 column */
    .stat-grid {
        grid-template-columns: 1fr !important;
    }

    /* Feature cards: handled by Streamlit columns collapsing naturally */

    /* Upload card */
    .upload-inner { padding: 24px 16px 20px; }
    .upload-icon-wrap { width: 52px; height: 52px; border-radius: 14px; }
    [data-testid="stFileUploader"] {
        margin-top: -150px !important;
        height: 150px !important;
    }

    /* Glass cards */
    .glass-card { padding: 18px; border-radius: 14px; }
    .feature-card { padding: 22px 16px; border-radius: 14px; }

    /* Score badge */
    .score-badge { width: 44px; height: 44px; font-size: 14px; }

    /* Section heading */
    .section-heading { font-size: 14px; }
}

@media (max-width: 480px) {
    .block-container {
        padding: 0.75rem 0.75rem 2rem !important;
    }
    .unified-header {
        padding: 8px 12px;
        border-radius: 12px;
    }
    .navbar-logo { width: 28px; height: 28px; font-size: 10px; border-radius: 8px; }
    .navbar-title { font-size: 14px; }
    .step { font-size: 11px; }
    .step-dot { width: 20px; height: 20px; font-size: 9px; }

    .fade-up h1,
    .fade-up [style*="font-size:42px"] {
        font-size: 24px !important;
    }
    .stat-value { font-size: 24px; }
    .stat-label { font-size: 10px; }
    .tag { font-size: 10px; padding: 3px 8px; }
}

</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.stage == "upload":

    _step_indicator(1)

    # Animated hero with gradient text
    st.markdown("""
    <div class="fade-up" style="text-align:center; padding: 48px 0 12px;">
        <h1 style="font-size:42px; font-weight:900; letter-spacing:-0.045em; margin-bottom:12px; line-height:1.1;">
            Find the <span class="gradient-text">perfect reviewers</span><br>for your research
        </h1>
        <p style="font-size:16px; color:#64748b; max-width:500px; margin:0 auto; line-height:1.75;">
            Upload your manuscript and our AI matches you with expert
            peer reviewers based on semantic understanding of your work.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # Upload card
    st.markdown("""
    <div class="upload-card fade-up-d1">
        <div class="upload-glow"></div>
        <div class="upload-inner">
            <div class="upload-icon-wrap">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <div style="font-size:17px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">Upload your manuscript</div>
            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">PDF, DOCX, or TXT — we'll extract title, abstract & keywords automatically</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload_hidden_label",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Extracting title, abstract, and keywords..."):
            fname = uploaded.name.lower()
            if fname.endswith(".pdf"):
                raw = extract_text_from_pdf(uploaded)
            elif fname.endswith((".docx", ".doc")):
                raw = extract_text_from_docx(uploaded)
            else:
                raw = uploaded.read().decode("utf-8", errors="ignore")
            st.session_state.parsed = parse_manuscript(raw)
            st.session_state.stage = "review"
            st.rerun()

    # Feature cards
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("""
        <div class="feature-card fade-up-d1">
            <div class="feature-icon" style="background:rgba(139,92,246,0.1);">🧠</div>
            <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">Semantic Matching</div>
            <div style="font-size:13px;color:#64748b;line-height:1.65;">AI understands your research context to find reviewers with deep domain expertise</div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="feature-card fade-up-d2">
            <div class="feature-icon" style="background:rgba(59,130,246,0.1);">📊</div>
            <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">Multi-Signal Scoring</div>
            <div style="font-size:13px;color:#64748b;line-height:1.65;">Topic, methodology, seniority, and recency combined into a single relevance score</div>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
        <div class="feature-card fade-up-d3">
            <div class="feature-icon" style="background:rgba(16,185,129,0.1);">✉️</div>
            <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">One-Click Invites</div>
            <div style="font-size:13px;color:#64748b;line-height:1.65;">Send peer review invitations directly with pre-filled professional email templates</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2 — REVIEW & CONFIGURE
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "review":
    parsed = st.session_state.parsed

    _step_indicator(2)

    if st.button("← Back"):
        st.session_state.stage = "upload"
        st.session_state.parsed = {}
        st.rerun()

    st.markdown("""
    <div class="fade-up">
        <h2 style="font-size:28px;font-weight:900;margin:4px 0 2px;letter-spacing:-0.03em;">Review & Configure</h2>
        <p style="color:#64748b;font-size:14px;margin-bottom:22px;">Verify extracted details and tune your search parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📝 Manuscript Details</div>', unsafe_allow_html=True)

    title = st.text_input("Title", value=parsed.get("title", ""))
    abstract = st.text_area("Abstract", value=parsed.get("abstract", ""), height=180)
    keywords_str = st.text_input("Keywords (comma-separated)", value=", ".join(parsed.get("keywords", [])))

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">⚙️ Search Settings</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        num_reviewers = st.slider("Number of reviewers", 3, 30, 10)
    with c2:
        num_candidates = st.slider("Search depth (candidates)", 20, 100, 50)

    st.markdown('<div class="section-heading">🛡️ Conflict of Interest — Excluded Authors</div>', unsafe_allow_html=True)

    # Pre-fill with extracted authors from the manuscript
    extracted_authors = parsed.get("authors", [])
    extracted_institutions = parsed.get("author_institutions", [])

    if extracted_authors:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2);
                    border-radius:12px; padding:14px 18px; margin-bottom:14px;">
            <div style="font-size:13px;font-weight:700;color:#f87171;margin-bottom:8px;">
                Authors extracted from manuscript (will be excluded from results):
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;">
                {''.join(f'<span style="background:rgba(239,68,68,0.15);color:#fca5a5;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">{esc(a)}</span>' for a in extracted_authors)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    author_names_str = st.text_input(
        "Excluded author names (comma-separated)",
        value=", ".join(extracted_authors),
        help="These authors will be excluded from the reviewer results. Edit or add more names.",
    )

    author_institutions_str = st.text_input(
        "Excluded institutions (comma-separated)",
        value=", ".join(extracted_institutions),
        help="Reviewers from these institutions will be flagged as COI.",
    )

    additional_exclude = st.text_area(
        "Additional authors to exclude (one per line)",
        height=80,
        help="Add any other names you want to exclude from the results, e.g. known collaborators.",
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Find Reviewers", type="primary", use_container_width=True):
        if not title or not abstract:
            st.error("Please provide both a title and an abstract.")
        else:
            kw = [k.strip() for k in keywords_str.split(",") if k.strip()]
            an = [n.strip() for n in author_names_str.split(",") if n.strip()] if author_names_str else []
            # Add additional exclusions
            if additional_exclude:
                an += [n.strip() for n in additional_exclude.split("\n") if n.strip()]
            an = list(dict.fromkeys(an))  # deduplicate
            ai = [i.strip() for i in author_institutions_str.split(",") if i.strip()] if author_institutions_str else []

            progress = st.progress(0, text="Initializing search pipeline...")
            try:
                from services.search_service import find_reviewers
                progress.progress(15, text="Generating query embeddings...")
                time.sleep(0.15)
                progress.progress(35, text="Searching vector database...")
                results = find_reviewers(
                    title=title, abstract=abstract, keywords=kw,
                    author_names=an, author_institutions=ai,
                    num_reviewers=num_reviewers, num_vector_candidates=num_candidates,
                )
                progress.progress(75, text="Scoring and ranking candidates...")
                time.sleep(0.15)
                progress.progress(90, text="Enriching contact information...")
                time.sleep(0.15)
                progress.progress(100, text="Complete!")
                time.sleep(0.3)
                progress.empty()
                st.session_state.results = results
                st.session_state.stage = "results"
                st.rerun()
            except Exception as e:
                progress.empty()
                st.error(f"Search failed: {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":
    results = st.session_state.results
    if not results:
        st.session_state.stage = "upload"
        st.rerun()

    meta = results.get("metadata", {})
    reviewers = results.get("reviewers", [])
    topics = results.get("extracted_topics", {})

    _step_indicator(3)

    if st.button("← New Search"):
        st.session_state.stage = "upload"
        st.session_state.parsed = {}
        st.session_state.results = None
        st.session_state.invite_open = {}
        st.session_state.invite_sent = {}
        st.rerun()

    st.markdown(f"""
    <div class="fade-up">
        <h2 style="font-size:28px;font-weight:900;margin:4px 0 2px;letter-spacing:-0.03em;">
            Reviewer Recommendations
        </h2>
        <p style="color:#64748b;font-size:14px;margin-bottom:22px;">
            <span class="gradient-text" style="font-weight:700;">{len(reviewers)}</span> top matches ranked by AI relevance scoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div class="stat-grid fade-up-d1">
        <div class="stat-card">
            <div class="stat-value">{meta.get('vector_candidates',0)}</div>
            <div class="stat-label">Candidates Searched</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{meta.get('reranked_candidates',0)}</div>
            <div class="stat-label">Scored & Ranked</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(reviewers)}</div>
            <div class="stat-label">Recommended</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Research profile
    if topics:
        tags_html = ""
        for d in topics.get("primary_domains", []):
            tags_html += f'<span class="tag tag-purple">{esc(d)}</span>'
        for m in topics.get("methodologies", []):
            tags_html += f'<span class="tag tag-blue">{esc(m)}</span>'
        for s in topics.get("sub_topics", []):
            tags_html += f'<span class="tag tag-green">{esc(s)}</span>'

        st.markdown(f"""
        <div class="glass-card fade-up-d2">
            <div class="section-label">Research Profile</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Reviewer Cards ────────────────────────────────────────────────────

    if not reviewers:
        st.warning("No matching reviewers found. Try broadening your keywords or increasing search depth.")
    else:
        for idx, r in enumerate(reviewers):
            score = r.get("overall_score", 0)
            sc_bg = "rgba(16,185,129,0.12)" if score >= 7 else "rgba(245,158,11,0.12)" if score >= 5 else "rgba(239,68,68,0.12)"
            sc_color = "#34d399" if score >= 7 else "#fbbf24" if score >= 5 else "#f87171"
            sc_border = "rgba(16,185,129,0.25)" if score >= 7 else "rgba(245,158,11,0.25)" if score >= 5 else "rgba(239,68,68,0.25)"

            name = esc(r.get("name", "Unknown"))
            inst = esc(r.get("institution", ""))
            rank = r.get("rank", "?")
            reason = esc(r.get("reasoning", ""))
            contact = r.get("contact", {})
            email = contact.get("email", "")
            email_is_inferred = contact.get("email_is_inferred", False)

            # Gradient divider
            if idx > 0:
                st.markdown('<div class="reviewer-divider"></div>', unsafe_allow_html=True)

            # Layout: rank | info | score bars
            col_rank, col_info, col_scores = st.columns([0.5, 4, 1.8])

            with col_rank:
                st.markdown(f"""<div style="text-align:center;padding-top:8px;">
                    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Rank</div>
                    <div style="font-size:24px;font-weight:900;letter-spacing:-0.03em;margin:4px 0 10px;" class="gradient-text">#{rank}</div>
                    <div class="score-badge" style="background:{sc_bg};color:{sc_color};border:2px solid {sc_border};">{score:.1f}</div>
                </div>""", unsafe_allow_html=True)

            with col_info:
                # Name
                st.markdown(f'<div style="font-size:18px;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">{name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:13px;color:#64748b;margin-bottom:10px;">📍 {inst or "Institution not listed"}</div>', unsafe_allow_html=True)

                # Topic tags
                rtags = "".join(
                    f'<span class="tag tag-purple tag-sm">{esc(t)}</span>'
                    for t in r.get("topics", [])[:5]
                )
                if rtags:
                    st.markdown(rtags, unsafe_allow_html=True)

                # Metrics
                h = r.get('h_index', 'N/A')
                cit = r.get('citation_count', 'N/A')
                wrk = r.get('works_count', 'N/A')
                st.markdown(f"""<div style="margin-top:10px;font-size:13px;color:#64748b;display:flex;gap:20px;flex-wrap:wrap;">
                    <span>H-index <b style="color:#cbd5e1;">{h}</b></span>
                    <span>Citations <b style="color:#cbd5e1;">{cit}</b></span>
                    <span>Works <b style="color:#cbd5e1;">{wrk}</b></span>
                </div>""", unsafe_allow_html=True)

                # AI Reasoning
                if reason:
                    st.markdown(f"""<div style="padding:10px 14px;background:rgba(99,102,241,0.06);
                        border-left:3px solid #818cf8;border-radius:0 12px 12px 0;font-size:13px;color:#94a3b8;
                        font-style:italic;margin-top:12px;line-height:1.7;">{reason}</div>""", unsafe_allow_html=True)

                # COI
                for flag in r.get("coi_flags", []):
                    sev = flag.get("severity", "medium")
                    coi_bg = "rgba(239,68,68,0.1)" if sev in ("high","critical") else "rgba(245,158,11,0.1)"
                    coi_color = "#f87171" if sev in ("high","critical") else "#fbbf24"
                    coi_border = "rgba(239,68,68,0.2)" if sev in ("high","critical") else "rgba(245,158,11,0.2)"
                    st.markdown(f'<div class="coi-flag" style="background:{coi_bg};color:{coi_color};border:1px solid {coi_border};">⚠️ {esc(flag.get("detail",""))}</div>', unsafe_allow_html=True)

                # Contact — email row
                if email:
                    if email_is_inferred:
                        ai_badge = ' <span class="ai-inferred-badge">AI INFERRED</span>'
                    else:
                        ai_badge = ""
                    email_html = (
                        '<div style="display:flex;align-items:center;gap:10px;margin-top:14px;flex-wrap:wrap;">'
                        '<span style="font-size:13px;">\u2709\uFE0F</span>'
                        f'<a href="mailto:{esc(email)}" style="font-size:13px;font-weight:700;color:#a5b4fc;text-decoration:none;border-bottom:1px dashed rgba(99,102,241,0.3);">{esc(email)}</a>'
                        f'{ai_badge}'
                        '</div>'
                    )
                    st.markdown(email_html, unsafe_allow_html=True)

                # Contact — profile links (separate st.markdown to avoid sanitizer issues)
                link_items = []
                for lbl, key in [("ORCID","orcid_url"),("OpenAlex","openalex_url"),("Homepage","homepage")]:
                    url = contact.get(key,"")
                    if url:
                        link_items.append(f'<a href="{esc(url)}" target="_blank" class="profile-link">{lbl}</a>')
                if link_items:
                    links_html = " ".join(link_items)
                    st.markdown(f'<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">{links_html}</div>', unsafe_allow_html=True)

            with col_scores:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                _score_bars(r)

            # ── Invite ────────────────────────────────────────────────────
            invite_key = f"invite_{rank}"
            already_sent = st.session_state.invite_sent.get(invite_key, False)

            if email and not already_sent:
                inv_c1, inv_c2 = st.columns([5, 1])
                with inv_c2:
                    if st.button("📨 Invite", key=f"btn_{invite_key}"):
                        st.session_state.invite_open[invite_key] = not st.session_state.invite_open.get(invite_key, False)
                        st.rerun()
            elif already_sent:
                st.markdown('<div style="text-align:right;padding:2px 0;"><span class="sent-badge">✓ Invite Sent</span></div>', unsafe_allow_html=True)

            if st.session_state.invite_open.get(invite_key, False) and email:
                paper_title = st.session_state.parsed.get("title", "our manuscript")
                default_subject = f"Invitation to Review: {paper_title[:80]}"
                reviewer_topics = r.get('topics', ['this field'])[:3]
                default_body = (
                    f"Dear {r.get('name', 'Professor')},\n\n"
                    f"We would like to invite you to serve as a peer reviewer for our manuscript titled "
                    f"\"{paper_title}\".\n\n"
                    f"Your expertise in {', '.join(reviewer_topics)} makes you an ideal "
                    f"candidate for reviewing this work. We believe your insights would greatly strengthen "
                    f"the quality of our research.\n\n"
                    f"Please let us know if you would be available and willing to review this manuscript. "
                    f"We would be happy to provide additional details about the submission.\n\n"
                    f"Thank you for considering this request.\n\n"
                    f"Best regards"
                )

                inferred_note = ""
                if email_is_inferred:
                    inferred_note = '<div style="margin-top:6px;padding:6px 10px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:8px;font-size:11px;color:#fbbf24;">⚠️ This email was AI-inferred. Please verify before sending.</div>'
                invite_header_html = (
                    '<div class="invite-header">'
                    '<div style="display:flex;align-items:center;gap:8px;">'
                    '<span style="font-size:14px;">\U0001F4E8</span>'
                    '<span style="font-size:14px;font-weight:700;color:#a5b4fc;">Compose Invite</span>'
                    '<span style="color:#475569;">\u2192</span>'
                    f'<span style="font-size:13px;font-weight:600;color:#e2e8f0;">{esc(email)}</span>'
                    '</div>'
                    f'{inferred_note}'
                    '</div>'
                )
                st.markdown(invite_header_html, unsafe_allow_html=True)

                subj = st.text_input("Subject", value=default_subject, key=f"subj_{invite_key}")
                body = st.text_area("Message", value=default_body, height=220, key=f"body_{invite_key}")

                send_cols = st.columns([3, 1, 1])
                with send_cols[1]:
                    if st.button("Send ✉️", key=f"send_{invite_key}", type="primary"):
                        st.session_state.invite_sent[invite_key] = True
                        st.session_state.invite_open[invite_key] = False
                        st.rerun()
                with send_cols[2]:
                    if st.button("Cancel", key=f"cancel_{invite_key}"):
                        st.session_state.invite_open[invite_key] = False
                        st.rerun()

    # ── Export ────────────────────────────────────────────────────────────
    if reviewers:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        export = []
        for r in reviewers:
            c = r.get("contact", {})
            email_val = c.get("email", "")
            email_note = " (AI inferred)" if c.get("email_is_inferred") else ""
            export.append({
                "Rank": r.get("rank"), "Name": r.get("name"),
                "Institution": r.get("institution", ""),
                "Email": f"{email_val}{email_note}" if email_val else "",
                "Score": round(r.get("overall_score", 0), 1),
                "H-Index": r.get("h_index", ""), "Citations": r.get("citation_count", ""),
                "ORCID": c.get("orcid_url", ""), "OpenAlex": c.get("openalex_url", ""),
            })
        df = pd.DataFrame(export)

        st.markdown('<div class="section-label">Export Results</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns([1, 1, 3])
        with ec1:
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                file_name=f"reviewers_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        with ec2:
            st.download_button("📊 Download Excel", _to_excel(df),
                file_name=f"reviewers_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
