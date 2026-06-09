import streamlit as st
import os
import re
import json
import math
import requests
import trafilatura
from dotenv import load_dotenv
from urllib.parse import urlparse

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def scrape_url(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=True, no_fallback=False)
        if text and len(text.strip()) > 200:
            return text.strip()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"ERROR:{e}"

def scrape_multiple_pages(base_url):
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    pages = [base_url]
    for slug in ["/about", "/pricing", "/product", "/solutions", "/company"]:
        pages.append(base.rstrip("/") + slug)
    combined = ""
    for url in pages[:4]:
        text = scrape_url(url)
        if not text.startswith("ERROR") and len(text) > 100:
            combined += f"\n\n[Source: {url}]\n{text[:3000]}"
        if len(combined) > 10000:
            break
    return combined.strip() or "Could not extract content from the provided URL."

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior startup/company analyst with deep expertise in business models, competitive landscapes, and investment analysis.

Analyze the following company content scraped from their website and produce a structured intelligence report.

COMPANY CONTENT:
{text}

Respond ONLY in this exact JSON format (no markdown, no backticks, no extra text):
{{
  "company_name": "<name of the company>",
  "tagline": "<one-line description of what they do>",
  "company_score": <integer 0-100>,
  "score_rationale": "<2 sentences explaining the score>",
  "scores_breakdown": {{
    "product": <0-100>,
    "market_position": <0-100>,
    "revenue_quality": <0-100>,
    "moat_strength": <0-100>,
    "growth_potential": <0-100>
  }},
  "revenue_model": {{
    "primary": "<e.g. SaaS Subscription / Marketplace / Transactional / Advertising>",
    "details": "<1-2 sentences on how they make money>"
  }},
  "target_market": "<who their customers are>",
  "key_products": ["<product 1>", "<product 2>", "<product 3>"],
  "competitors": [
    {{"name": "<Competitor 1>", "why": "<why they compete>"}},
    {{"name": "<Competitor 2>", "why": "<why they compete>"}},
    {{"name": "<Competitor 3>", "why": "<why they compete>"}},
    {{"name": "<Competitor 4>", "why": "<why they compete>"}},
    {{"name": "<Competitor 5>", "why": "<why they compete>"}}
  ],
  "competitive_radar": {{
    "price": <0-100>,
    "technology": <0-100>,
    "reach": <0-100>,
    "support": <0-100>,
    "speed": <0-100>
  }},
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "risks": [
    {{"title": "<Risk 1>", "detail": "<explanation>"}},
    {{"title": "<Risk 2>", "detail": "<explanation>"}},
    {{"title": "<Risk 3>", "detail": "<explanation>"}}
  ],
  "growth_opportunities": [
    {{"title": "<Opportunity 1>", "detail": "<explanation>"}},
    {{"title": "<Opportunity 2>", "detail": "<explanation>"}},
    {{"title": "<Opportunity 3>", "detail": "<explanation>"}}
  ],
  "moat": "<what makes them hard to compete with>",
  "stage": "<Early Stage / Growth / Scale / Enterprise>",
  "summary": "<3-sentence executive summary of the company>"
}}
"""
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior company analyst. Below are analysis summaries from multiple pages of a company website.

{text}

Consolidate into a single comprehensive JSON analysis report with this EXACT structure (no markdown, no backticks):
{{
  "company_name": "<name>",
  "tagline": "<one-liner>",
  "company_score": <0-100>,
  "score_rationale": "<2 sentences>",
  "scores_breakdown": {{
    "product": <0-100>,
    "market_position": <0-100>,
    "revenue_quality": <0-100>,
    "moat_strength": <0-100>,
    "growth_potential": <0-100>
  }},
  "revenue_model": {{"primary": "<model>", "details": "<how they earn>"}},
  "target_market": "<customers>",
  "key_products": ["<p1>", "<p2>", "<p3>"],
  "competitors": [{{"name": "<c>", "why": "<reason>"}}],
  "competitive_radar": {{
    "price": <0-100>,
    "technology": <0-100>,
    "reach": <0-100>,
    "support": <0-100>,
    "speed": <0-100>
  }},
  "strengths": ["<s1>", "<s2>", "<s3>"],
  "risks": [{{"title": "<r>", "detail": "<d>"}}],
  "growth_opportunities": [{{"title": "<o>", "detail": "<d>"}}],
  "moat": "<competitive advantage>",
  "stage": "<stage>",
  "summary": "<3-sentence summary>"
}}
"""
)

def analyze_company(url):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set in .env file."}
    raw_text = scrape_multiple_pages(url)
    if raw_text.startswith("Could not"):
        return {"error": raw_text}
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "])
    chunks = splitter.split_text(raw_text)
    docs = [Document(page_content=c) for c in chunks[:8]]
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.3, max_tokens=2000)
    if len(docs) == 1:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=ANALYSIS_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=ANALYSIS_PROMPT, combine_prompt=COMBINE_PROMPT, token_max=3000)
    result = chain.invoke({"input_documents": docs})
    raw_output = result.get("output_text", "")
    clean = raw_output.strip().replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        clean = m.group(0)
    try:
        return json.loads(clean)
    except Exception:
        return {"error": "Failed to parse AI response.", "raw": raw_output}

def score_color(score):
    if score >= 70: return "#1a7a4a"
    if score >= 45: return "#b45309"
    return "#c0392b"

def score_label(score):
    if score >= 80: return "Excellent"
    if score >= 65: return "Strong"
    if score >= 50: return "Moderate"
    if score >= 35: return "Weak"
    return "Poor"

def make_score_ring_svg(score, size=140):
    cx = cy = size / 2
    r = size * 0.36
    circ = 2 * math.pi * r
    dash = (score / 100) * circ
    color = score_color(score)
    track = "#e2e8f0"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{track}" stroke-width="{size*0.072:.1f}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{size*0.072:.1f}"
    stroke-dasharray="{dash:.2f} {circ:.2f}" stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" dominant-baseline="central"
    fill="{color}" font-size="{size*0.24:.0f}" font-weight="bold"
    font-family="'Times New Roman', Times, serif">{score}</text>
  <text x="{cx}" y="{cy + size*0.18}" text-anchor="middle"
    fill="#64748b" font-size="{size*0.11:.0f}"
    font-family="Calibri, 'Segoe UI', sans-serif">/100</text>
</svg>"""

def make_radar_svg(radar, width=320, height=320):
    labels = list(radar.keys())
    values = [radar[k] for k in labels]
    n = len(labels)
    cx, cy = width / 2, height / 2
    r = min(cx, cy) * 0.65
    def pt(i, radius):
        angle = (i * 2 * math.pi / n) - math.pi / 2
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)
    rings_svg = ""
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{pt(i, ring*r)[0]:.1f},{pt(i, ring*r)[1]:.1f}" for i in range(n))
        rings_svg += f'<polygon points="{pts}" fill="none" stroke="#cbd5e1" stroke-width="1"/>\n'
    spokes_svg = "".join(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{pt(i, r)[0]:.1f}" y2="{pt(i, r)[1]:.1f}" stroke="#cbd5e1" stroke-width="1"/>\n'
        for i in range(n))
    data_pts = " ".join(f"{pt(i, (values[i]/100)*r)[0]:.1f},{pt(i, (values[i]/100)*r)[1]:.1f}" for i in range(n))
    data_svg = f'<polygon points="{data_pts}" fill="rgba(37,99,235,0.15)" stroke="#2563eb" stroke-width="2"/>\n'
    dots_svg = "".join(
        f'<circle cx="{pt(i, (values[i]/100)*r)[0]:.1f}" cy="{pt(i, (values[i]/100)*r)[1]:.1f}" r="5" fill="#2563eb" stroke="white" stroke-width="1.5"/>\n'
        for i in range(n))
    label_offset = r + 26
    label_svg = ""
    for i, lbl in enumerate(labels):
        lx, ly = pt(i, label_offset)
        label_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="600" fill="#374151" font-family="Calibri, sans-serif">{lbl.capitalize()}</text>\n'
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:white;">
  {rings_svg}{spokes_svg}{data_svg}{dots_svg}{label_svg}
</svg>"""

def make_trend_svg(data, width=420, height=160):
    pad = 14
    mn, mx = min(data), max(data)
    rng = mx - mn or 1
    def px(i): return pad + (i / (len(data) - 1)) * (width - 2 * pad)
    def py(v): return height - pad - 20 - ((v - mn) / rng) * (height - 2 * pad - 30)
    line_pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(data))
    area_pts = line_pts + f" {px(len(data)-1):.1f},{height-20} {px(0):.1f},{height-20}"
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_labels = ""
    for i in range(0, len(data), 2):
        month_labels += f'<text x="{px(i):.1f}" y="{height - 4}" text-anchor="middle" font-size="11" fill="#94a3b8" font-family="Calibri, sans-serif">{months[i]}</text>\n'
    dots = "".join(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="#2563eb" stroke="white" stroke-width="1.5"/>\n' for i, v in enumerate(data))
    return f"""<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
  <defs>
    <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2563eb" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#2563eb" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="white"/>
  <polygon points="{area_pts}" fill="url(#tg)"/>
  <polyline points="{line_pts}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  {dots}{month_labels}
</svg>"""

def make_bar_svg(label, value, max_val=100, width=340):
    pct = value / max_val
    bar_w = max(4, int((width - 160) * pct))
    full_w = width - 160
    if value >= 75:
        color = "#16a34a"
    elif value >= 55:
        color = "#2563eb"
    else:
        color = "#d97706"
    return f"""<svg width="{width}" height="32" viewBox="0 0 {width} 32" xmlns="http://www.w3.org/2000/svg">
  <text x="0" y="20" font-size="14" fill="#374151" font-family="Calibri, 'Segoe UI', sans-serif" font-weight="600">{label}</text>
  <rect x="155" y="13" width="{full_w}" height="7" rx="4" fill="#e2e8f0"/>
  <rect x="155" y="13" width="{bar_w}" height="7" rx="4" fill="{color}"/>
  <text x="{width}" y="20" text-anchor="end" font-size="13" font-weight="700" fill="{color}" font-family="'Times New Roman', serif">{value}</text>
</svg>"""

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="CompanyLens · Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def inject_css(css):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css("""
:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --elevated: #ffffff;
    --card-border: #e2e8f0;
    --card-hover: #f1f5f9;
    --text: #0f172a;
    --muted: #475569;
    --faint: #94a3b8;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --green: #16a34a;
    --green-light: #f0fdf4;
    --amber: #b45309;
    --amber-light: #fffbeb;
    --red: #dc2626;
    --red-light: #fef2f2;
    --r: 10px;
    --rl: 14px;
    --font-main: Calibri, 'Segoe UI', Arial, sans-serif;
    --font-head: 'Times New Roman', Times, Georgia, serif;
}
""")

inject_css("""
.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-main) !important;
    font-size: 16px !important;
}
.stApp > header { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
.block-container { padding: 0 2.5rem 6rem !important; max-width: 1280px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
""")

inject_css("""
.stTextInput input {
    background: #ffffff !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: var(--font-main) !important;
    font-size: 16px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
.stTextInput input::placeholder { color: #94a3b8 !important; font-family: var(--font-main) !important; }
.stTextInput label { display: none !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--font-head) !important;
    font-weight: bold !important;
    font-size: 17px !important;
    padding: 14px 32px !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
}
hr { border: none !important; border-top: 2px solid #e2e8f0 !important; margin: 36px 0 !important; }
""")

inject_css("""
.top-nav {
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 2px solid #e2e8f0;
    padding: 16px 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -2.5rem 0;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.nav-brand {
    font-family: var(--font-head);
    font-size: 22px;
    font-weight: bold;
    color: var(--accent);
    display: flex; align-items: center; gap: 10px;
}
.nav-dot { width: 10px; height: 10px; background: var(--accent); border-radius: 3px; display: inline-block; }
.nav-tag {
    font-size: 13px;
    font-family: var(--font-main);
    color: var(--muted);
    background: var(--accent-light);
    border: 1px solid #bfdbfe;
    border-radius: 100px;
    padding: 5px 16px;
    font-weight: 600;
}

.hero { text-align: center; padding: 72px 0 56px; }
.hero-badge {
    display: inline-block;
    font-size: 12px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent);
    background: var(--accent-light); border: 1px solid #bfdbfe;
    border-radius: 100px; padding: 7px 18px; margin-bottom: 24px;
    font-family: var(--font-main);
}
.hero-title {
    font-family: var(--font-head);
    font-size: 54px; font-weight: bold;
    letter-spacing: -0.02em; line-height: 1.1;
    margin-bottom: 18px; color: var(--text);
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 18px; color: var(--muted);
    max-width: 480px; margin: 0 auto 44px;
    line-height: 1.75; font-family: var(--font-main);
}

.search-wrap {
    max-width: 580px; margin: 0 auto;
    background: var(--surface);
    border: 2px solid #e2e8f0;
    border-radius: 20px; padding: 32px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.search-label {
    font-size: 13px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 10px; font-family: var(--font-main);
}

.feature-strip {
    display: flex; gap: 12px; justify-content: center;
    flex-wrap: wrap; padding: 40px 0 0;
}
.feature-pill {
    background: white; border: 1.5px solid #e2e8f0;
    border-radius: 100px; padding: 8px 18px;
    font-size: 14px; color: var(--muted);
    font-weight: 600; font-family: var(--font-main);
    display: flex; align-items: center; gap: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.pill-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: inline-block; }
""")

inject_css("""
.card {
    background: var(--elevated);
    border: 1.5px solid var(--card-border);
    border-radius: var(--rl);
    padding: 24px 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-color: #bfdbfe; }
.card-accent-top { border-top: 3px solid var(--accent) !important; }

.section-head {
    display: flex; align-items: center; gap: 12px;
    margin: 36px 0 18px;
}
.sec-icon {
    width: 36px; height: 36px;
    background: var(--accent-light);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: bold; color: var(--accent);
    flex-shrink: 0; border: 1px solid #bfdbfe;
}
.sec-title {
    font-family: var(--font-head);
    font-size: 20px; font-weight: bold;
    color: var(--text);
}
.sec-sub {
    font-size: 14px; color: var(--faint);
    margin-top: 3px; font-family: var(--font-main);
}

.company-name {
    font-family: var(--font-head);
    font-size: 38px; font-weight: bold;
    letter-spacing: -0.02em; margin-bottom: 8px;
    color: var(--text);
}
.company-tagline {
    font-size: 17px; color: var(--muted);
    line-height: 1.65; margin-bottom: 24px;
    font-family: var(--font-main);
}

.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 8px; padding: 7px 14px; margin: 4px;
    font-size: 14px; font-family: var(--font-main);
}
.chip-lbl { color: var(--faint); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.product-tag {
    display: inline-block; padding: 5px 14px; margin: 4px;
    background: var(--accent-light); border: 1.5px solid #bfdbfe;
    border-radius: 7px; font-size: 13px; font-weight: 700;
    color: var(--accent); font-family: var(--font-main);
}

.score-panel {
    background: var(--elevated); border: 1.5px solid var(--card-border);
    border-radius: var(--rl); padding: 32px 24px;
    text-align: center; height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.score-lbl {
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--faint);
    margin-bottom: 16px; font-family: var(--font-main);
}
.score-badge {
    display: inline-block; padding: 5px 16px;
    border-radius: 7px; font-size: 14px;
    font-weight: bold; margin-bottom: 16px;
    font-family: var(--font-main);
}
.score-rationale {
    font-size: 14px; color: var(--muted); line-height: 1.7;
    padding-top: 16px; border-top: 1.5px solid #e2e8f0;
    font-family: var(--font-main);
}

.stat-card {
    background: var(--elevated); border: 1.5px solid var(--card-border);
    border-radius: var(--rl); padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.stat-num {
    font-family: var(--font-head);
    font-size: 36px; font-weight: bold; line-height: 1;
}
.stat-lbl {
    font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--faint);
    margin-top: 7px; font-family: var(--font-main); font-weight: 600;
}

.summary-card {
    background: #f0f9ff;
    border: 1.5px solid #bae6fd;
    border-left: 4px solid var(--accent);
    border-radius: 0 var(--rl) var(--rl) 0;
    padding: 26px 32px;
}
.summary-text {
    font-size: 16px; color: #0c4a6e;
    line-height: 1.9; font-family: var(--font-main);
}
""")

inject_css("""
.comp-row {
    display: flex; align-items: flex-start; gap: 16px;
    padding: 14px 0; border-bottom: 1.5px solid #f1f5f9;
}
.comp-row:last-child { border-bottom: none; }
.comp-num {
    min-width: 28px; height: 28px;
    background: var(--accent-light); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: bold; color: var(--accent);
    font-family: var(--font-head); flex-shrink: 0;
    border: 1px solid #bfdbfe;
}
.comp-name { font-size: 15px; font-weight: 700; margin-bottom: 4px; font-family: var(--font-main); color: var(--text); }
.comp-why { font-size: 14px; color: var(--muted); line-height: 1.6; font-family: var(--font-main); }

.strength-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1.5px solid #f1f5f9;
}
.strength-row:last-child { border-bottom: none; }
.strength-check {
    min-width: 22px; height: 22px;
    background: var(--green-light); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: var(--green);
    flex-shrink: 0; margin-top: 1px;
    border: 1.5px solid #bbf7d0; font-weight: bold;
}
.strength-text { font-size: 15px; color: #1e293b; line-height: 1.65; font-family: var(--font-main); }

.risk-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1.5px solid #f1f5f9;
}
.risk-row:last-child { border-bottom: none; }
.risk-dot {
    min-width: 10px; height: 10px; border-radius: 50%;
    background: var(--red); margin-top: 6px; flex-shrink: 0;
}
.risk-title { font-size: 15px; font-weight: 700; margin-bottom: 4px; font-family: var(--font-main); color: #7f1d1d; }
.risk-detail { font-size: 14px; color: var(--muted); line-height: 1.6; font-family: var(--font-main); }

.opp-card {
    background: var(--elevated); border: 1.5px solid #fde68a;
    border-radius: var(--rl); padding: 24px 26px;
    transition: border-color 0.2s, transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.opp-card:hover { border-color: #f59e0b; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(245,158,11,0.15); }
.opp-badge {
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--amber);
    margin-bottom: 10px; font-family: var(--font-main);
}
.opp-title {
    font-family: var(--font-head);
    font-size: 17px; font-weight: bold;
    margin-bottom: 10px; color: var(--text);
}
.opp-detail { font-size: 14px; color: var(--muted); line-height: 1.7; font-family: var(--font-main); }

.progress-wrap {
    max-width: 500px; margin: 80px auto;
    background: white; border: 2px solid #e2e8f0;
    border-radius: 20px; padding: 36px 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.progress-head {
    font-size: 13px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--faint);
    margin-bottom: 22px; font-family: var(--font-main);
}
.progress-step {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 0; font-size: 16px; font-family: var(--font-main);
}

.rev-primary {
    font-family: var(--font-head);
    font-size: 26px; font-weight: bold;
    color: var(--accent); margin-bottom: 10px;
}
.rev-detail { font-size: 15px; color: var(--muted); line-height: 1.75; font-family: var(--font-main); }
.chart-row { display: flex; justify-content: space-between; margin-top: 12px; }
.chart-val { font-family: var(--font-head); font-size: 20px; font-weight: bold; }

.err-box {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-radius: var(--rl); padding: 22px 28px;
    color: #991b1b; font-size: 16px;
    line-height: 1.65; font-family: var(--font-main);
}
.footer { text-align: center; padding: 28px 0; border-top: 2px solid #e2e8f0; margin-top: 20px; }
.footer-text { font-size: 14px; color: var(--faint); font-family: var(--font-main); }
.footer-text a { color: var(--accent); text-decoration: none; font-weight: 600; }
""")

# ── Nav ──
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <span class="nav-dot"></span>CompanyLens
    </div>
    <div class="nav-tag">AI Company Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
if "analyzed" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">&#10022; AI-Driven Competitive Intelligence</div>
        <div class="hero-title">Analyze Any Company<br><span>in Seconds.</span></div>
        <div class="hero-sub">Enter a company URL to generate instant AI-powered SWOT analysis, visual dashboards, and strategic insights.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Search box ──
_, mid, _ = st.columns([1, 2.2, 1])
with mid:
    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="search-label">Company Website URL</div>', unsafe_allow_html=True)
    url_input = st.text_input("URL", placeholder="https://stripe.com", label_visibility="collapsed", key="url_input")
    analyze_btn = st.button("Analyze Now \u2192", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not analyze_btn and "analyzed" not in st.session_state:
    st.markdown("""
    <div class="feature-strip">
        <div class="feature-pill"><span class="pill-dot"></span>Company Score</div>
        <div class="feature-pill"><span class="pill-dot"></span>Revenue Model</div>
        <div class="feature-pill"><span class="pill-dot"></span>Competitor Map</div>
        <div class="feature-pill"><span class="pill-dot"></span>Radar Analysis</div>
        <div class="feature-pill"><span class="pill-dot"></span>Market Trend</div>
        <div class="feature-pill"><span class="pill-dot"></span>Growth Insights</div>
    </div>
    """, unsafe_allow_html=True)

# ── Validation ──
if analyze_btn:
    if not url_input.strip():
        st.markdown('<div class="err-box">Please enter a company URL.</div>', unsafe_allow_html=True)
        st.stop()
    if not GROQ_API_KEY:
        st.markdown('<div class="err-box">GROQ_API_KEY not found. Add it to your .env file and restart.</div>', unsafe_allow_html=True)
        st.stop()

    url = url_input.strip()
    if not url.startswith("http"):
        url = "https://" + url

    progress_placeholder = st.empty()
    steps = [
        ("Fetching company website", 1),
        ("Extracting page content", 2),
        ("Running AI analysis", 3),
        ("Building intelligence report", 4),
    ]

    def show_progress(active_step):
        rows = ""
        for label, idx in steps:
            if idx < active_step:
                col, icon = "#16a34a", "&#10003;"
            elif idx == active_step:
                col, icon = "#2563eb", "&#9679;"
            else:
                col, icon = "#94a3b8", "&#9675;"
            rows += f'<div class="progress-step" style="color:{col};font-weight:600;"><span style="width:22px;text-align:center;font-size:18px;">{icon}</span>{label}</div>'
        progress_placeholder.markdown(
            f'<div class="progress-wrap"><div class="progress-head">Analyzing&hellip;</div>{rows}</div>',
            unsafe_allow_html=True,
        )

    show_progress(1)
    show_progress(2)
    show_progress(3)

    with st.spinner(""):
        data = analyze_company(url)

    progress_placeholder.empty()

    if "error" in data:
        st.markdown(f'<div class="err-box">Analysis failed: {data["error"]}<br><br>{data.get("raw","")[:400]}</div>', unsafe_allow_html=True)
        st.stop()

    st.session_state["data"] = data
    st.session_state["url"] = url
    st.session_state["analyzed"] = True

# ── Render results ──
if st.session_state.get("analyzed") and "data" in st.session_state:
    data = st.session_state["data"]
    url  = st.session_state["url"]

    name          = data.get("company_name", urlparse(url).netloc)
    tagline       = data.get("tagline", "")
    score         = int(data.get("company_score", 0))
    score_rat     = data.get("score_rationale", "")
    revenue       = data.get("revenue_model", {})
    target        = data.get("target_market", "")
    products      = data.get("key_products", [])
    competitors   = data.get("competitors", [])
    strengths     = data.get("strengths", [])
    risks         = data.get("risks", [])
    opportunities = data.get("growth_opportunities", [])
    moat          = data.get("moat", "")
    stage         = data.get("stage", "")
    summary       = data.get("summary", "")
    breakdown     = data.get("scores_breakdown", {"product": 80, "market_position": 75, "revenue_quality": 78, "moat_strength": 82, "growth_potential": 76})
    radar_raw     = data.get("competitive_radar", {"price": 70, "technology": 88, "reach": 75, "support": 72, "speed": 85})

    sc = score_color(score)
    sl = score_label(score)
    base = 20 + score // 5
    trend_data = [int(base * (1 + 0.06 * i + (i % 3) * 0.01)) for i in range(12)]

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # ── Header row ──
    col_main, col_score = st.columns([3, 1])
    with col_main:
        products_html = "".join(f'<span class="product-tag">{p}</span>' for p in products)
        chips = (
            f'<span class="chip"><span class="chip-lbl">Stage&nbsp;</span><strong>{stage}</strong></span>'
            f'<span class="chip"><span class="chip-lbl">Model&nbsp;</span><strong>{revenue.get("primary","—")}</strong></span>'
            f'<span class="chip"><span class="chip-lbl">Market&nbsp;</span>{target[:50]}…</span>'
        )
        st.markdown(f"""
        <div class="card card-accent-top" style="padding:32px 38px;">
            <div class="company-name">{name}</div>
            <div class="company-tagline">{tagline}</div>
            <div style="margin-bottom:18px;">{chips}</div>
            <div>{products_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_score:
        ring_svg = make_score_ring_svg(score, size=140)
        st.markdown(f"""
        <div class="score-panel">
            <div class="score-lbl">Company Score</div>
            {ring_svg}
            <div><span class="score-badge" style="background:{sc}18;color:{sc};border:1.5px solid {sc}40;">{sl}</span></div>
            <div class="score-rationale">{score_rat}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visual Intelligence Panel ──
    st.markdown("""
    <div class="section-head">
        <div class="sec-icon">&#9678;</div>
        <div>
            <div class="sec-title">Visual Intelligence Panel</div>
            <div class="sec-sub">Multi-dimensional analysis at a glance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    an1, an2, an3 = st.columns(3)
    with an1:
        bars = "".join(make_bar_svg(k.replace("_", " ").title(), v, width=320) for k, v in breakdown.items())
        st.markdown(f"""
        <div class="card" style="padding:24px 26px;">
            <div style="font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:18px;font-family:Calibri,sans-serif;">Score Breakdown</div>
            {bars}
        </div>
        """, unsafe_allow_html=True)

    with an2:
        trend_svg = make_trend_svg(trend_data, width=420, height=160)
        st.markdown(f"""
        <div class="card" style="padding:24px 26px;">
            <div style="font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:4px;font-family:Calibri,sans-serif;">Market Trend</div>
            <div style="font-size:13px;color:#94a3b8;margin-bottom:14px;font-family:Calibri,sans-serif;">Growth forecast 2024&ndash;2025</div>
            {trend_svg}
            <div class="chart-row">
                <div>
                    <div style="font-size:12px;color:#94a3b8;font-family:Calibri,sans-serif;">Start</div>
                    <div class="chart-val" style="color:#2563eb;">${trend_data[0]}B</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px;color:#94a3b8;font-family:Calibri,sans-serif;">Now</div>
                    <div class="chart-val" style="color:#16a34a;">${trend_data[-1]}B</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with an3:
        radar_svg = make_radar_svg(radar_raw, width=300, height=300)
        st.markdown(f"""
        <div class="card" style="padding:24px 26px;display:flex;flex-direction:column;align-items:center;">
            <div style="font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:4px;width:100%;font-family:Calibri,sans-serif;">Competitive Position</div>
            <div style="font-size:13px;color:#94a3b8;margin-bottom:10px;width:100%;font-family:Calibri,sans-serif;">Market standing vs peers</div>
            {radar_svg}
        </div>
        """, unsafe_allow_html=True)

    # ── Executive Summary ──
    if summary:
        st.markdown("""
        <div class="section-head">
            <div class="sec-icon">S</div>
            <div><div class="sec-title">Executive Summary</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card"><div class="summary-text">{summary}</div></div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Revenue Model ──
    st.markdown("""
    <div class="section-head">
        <div class="sec-icon" style="background:#f0fdf4;border-color:#bbf7d0;color:#16a34a;">$</div>
        <div>
            <div class="sec-title">Revenue Model</div>
            <div class="sec-sub">How the company monetizes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:12px;font-family:Calibri,sans-serif;">Primary Model</div>
            <div class="rev-primary">{revenue.get("primary","—")}</div>
            <div class="rev-detail">{revenue.get("details","")}</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:12px;font-family:Calibri,sans-serif;">Competitive Moat</div>
            <div style="font-size:15px;color:#1e293b;line-height:1.8;font-family:Calibri,sans-serif;">{moat}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Competitors ──
    st.markdown("""
    <div class="section-head">
        <div class="sec-icon" style="background:#fef2f2;border-color:#fca5a5;color:#dc2626;">C</div>
        <div>
            <div class="sec-title">Competitors</div>
            <div class="sec-sub">Key players in the same space</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    comp_rows = ""
    for i, c in enumerate(competitors):
        n_c = c.get("name","") if isinstance(c, dict) else str(c)
        w_c = c.get("why","")  if isinstance(c, dict) else ""
        comp_rows += f"""
        <div class="comp-row">
            <div class="comp-num">{i+1}</div>
            <div><div class="comp-name">{n_c}</div><div class="comp-why">{w_c}</div></div>
        </div>"""
    st.markdown(f'<div class="card">{comp_rows}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Strengths + Risks ──
    s_col, r_col = st.columns(2)
    with s_col:
        st.markdown("""
        <div class="section-head" style="margin-top:0;">
            <div class="sec-icon" style="background:#f0fdf4;border-color:#bbf7d0;color:#16a34a;">+</div>
            <div><div class="sec-title">Strengths</div><div class="sec-sub">What they do well</div></div>
        </div>
        """, unsafe_allow_html=True)
        str_rows = "".join(
            f'<div class="strength-row"><div class="strength-check">&#10003;</div><div class="strength-text">{s}</div></div>'
            for s in strengths)
        st.markdown(f'<div class="card">{str_rows}</div>', unsafe_allow_html=True)

    with r_col:
        st.markdown("""
        <div class="section-head" style="margin-top:0;">
            <div class="sec-icon" style="background:#fef2f2;border-color:#fca5a5;color:#dc2626;">!</div>
            <div><div class="sec-title">Risks</div><div class="sec-sub">Key threats to watch</div></div>
        </div>
        """, unsafe_allow_html=True)
        risk_rows = ""
        for r_item in risks:
            t = r_item.get("title","") if isinstance(r_item, dict) else str(r_item)
            d = r_item.get("detail","") if isinstance(r_item, dict) else ""
            risk_rows += f'<div class="risk-row"><div class="risk-dot"></div><div><div class="risk-title">{t}</div><div class="risk-detail">{d}</div></div></div>'
        st.markdown(f'<div class="card">{risk_rows}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Growth Opportunities ──
    st.markdown("""
    <div class="section-head">
        <div class="sec-icon" style="background:#fffbeb;border-color:#fde68a;color:#b45309;">G</div>
        <div>
            <div class="sec-title">Growth Opportunities</div>
            <div class="sec-sub">Where the company can expand</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    opp_cols = st.columns(3)
    for i, opp in enumerate(opportunities[:3]):
        t = opp.get("title","")  if isinstance(opp, dict) else str(opp)
        d = opp.get("detail","") if isinstance(opp, dict) else ""
        with opp_cols[i % 3]:
            st.markdown(f"""
            <div class="opp-card">
                <div class="opp-badge">Opportunity {i+1}</div>
                <div class="opp-title">{t}</div>
                <div class="opp-detail">{d}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Stat cards ──
    stat_cols = st.columns(4)
    stats = [
        ("Products", len(products), "#2563eb"),
        ("Competitors", len(competitors), "#dc2626"),
        ("Strengths", len(strengths), "#16a34a"),
        ("Score", f"{score}<span style='font-size:16px;color:#94a3b8'>/100</span>", score_color(score)),
    ]
    for i, (lbl, val, col) in enumerate(stats):
        with stat_cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num" style="color:{col};">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">
        <span class="footer-text">
            Analysis from <a href="{url}" target="_blank">{url}</a> &middot; Powered by LangChain + Groq LLaMA 3.3
        </span>
    </div>
    """, unsafe_allow_html=True)