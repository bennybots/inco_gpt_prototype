# app.py — Incodox GPT (updated per Eran's feedback)
# Drop-in replacement: keeps your extractor + adds Summary of Approval (email style),
# supply-chain diagram, real dispute chat, mandatory About Us, fixed deliverables,
# and added fields (legal addresses, employees/roles).

# --- imports & env ---
import os, json, requests, datetime as dt, re, random
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
from typing import List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import streamlit as st
from dataclasses import dataclass, asdict, field
from jinja2 import Template

# your existing extractor (unchanged)
from app.ie.extractors import extract_from_chunk

print("DEBUG — Loaded key:", os.getenv("OPENAI_API_KEY"))
DEBUG = os.getenv("RETRIEVER_DEBUG", "0") == "1"

# --- streamlit config must be first ---
st.set_page_config(page_title="Incodox GPT - Prototype", page_icon="🧠", layout="wide")
st.title("Incodox GPT")

# =======================
# Data model (updated)
# =======================
@dataclass
class CaseFacts:
    # Legal entities
    full_legal_name_1: str = ""
    full_legal_name_2: str = ""
    entity1_address: str = ""      # NEW
    entity2_address: str = ""      # NEW

    # Jurisdictions
    countries_of_incorp: List[str] = field(default_factory=list)

    # People / roles
    entity1_employees: Optional[int] = None   # NEW
    entity2_employees: Optional[int] = None   # NEW
    entity1_roles: List[str] = field(default_factory=list)
    entity2_roles: List[str] = field(default_factory=list)

    # Business + IP
    business_model: str = ""
    role: str = ""
    ip_assets: str = ""
    transaction_flows: str = ""

    # Classification
    recommended_model: str = "TBD"
    confidence: Optional[float] = None

    # Suggestions
    ip_name_suggestion: List[str] = field(default_factory=list)


def _merge_facts(extracted: dict, edited: dict) -> CaseFacts:
    d = {**(extracted or {}), **(edited or {})}
    if isinstance(d.get("countries_of_incorp"), str):
        d["countries_of_incorp"] = [x.strip() for x in d["countries_of_incorp"].split(",") if x.strip()]
    return CaseFacts(
        full_legal_name_1=d.get("full_legal_name_1",""),
        full_legal_name_2=d.get("full_legal_name_2",""),
        entity1_address=d.get("entity1_address",""),
        entity2_address=d.get("entity2_address",""),
        countries_of_incorp=d.get("countries_of_incorp") or [],
        entity1_employees=d.get("entity1_employees"),
        entity2_employees=d.get("entity2_employees"),
        entity1_roles=d.get("entity1_roles") or [],
        entity2_roles=d.get("entity2_roles") or [],
        business_model=d.get("business_model",""),
        role=d.get("role",""),
        ip_assets=d.get("ip_assets",""),
        transaction_flows=d.get("transaction_flows",""),
        recommended_model=d.get("recommended_model","TBD"),
        confidence=d.get("confidence"),
        ip_name_suggestion=d.get("ip_name_suggestion") or [],
    )

# =======================
# Helpers
# =======================
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36 Edg/123",
]

def _parse_basic_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body_txt = " ".join(t.get_text(" ", strip=True) for t in soup.select("h1,h2,h3,p,li"))
    metas = []
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"): metas.append(md["content"])
    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"): metas.append(og["content"])
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return " ".join([title] + metas + [body_txt]).strip()

def safe_index(value, options, default=0):
    try:
        return options.index(value)
    except Exception:
        return default

def fetch_text_any(url: str, timeout: int = None) -> str:
    if not url: return ""
    timeout = timeout or int(os.getenv("ENRICH_TIMEOUT", "12"))
    uas = _UAS[:]
    random.shuffle(uas)
    for ua in uas:
        try:
            r = requests.get(
                url, timeout=timeout, allow_redirects=True,
                headers={
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                    "Cache-Control": "no-cache",
                },
            )
            if r.status_code == 200 and r.text:
                return _parse_basic_html(r.text)
        except Exception:
            pass
    try:
        p = urlparse(url)
        proxy_url = f"https://r.jina.ai/{p.scheme}://{p.netloc}{p.path or '/'}"
        r2 = requests.get(proxy_url, timeout=timeout)
        if r2.status_code == 200 and r2.text:
            return r2.text.strip()
    except Exception:
        pass
    return ""

def _dedupe_shrink(texts: list[str], max_chars: int = 20000) -> str:
    seen = set()
    lines = []
    for t in texts:
        for s in re.split(r"[\n\r]+", (t or "")):
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) < 40:
                continue
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            lines.append(s)
    combined = " ".join(lines)
    return combined[:max_chars]

def normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    try:
        p = urlparse(raw)
        if not p.netloc:
            return ""
        return raw.rstrip("/")
    except Exception:
        return ""

def fetch_site_text(url: str, timeout=12) -> str:
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36"
                )
            },
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        body_txt = " ".join(t.get_text(" ", strip=True) for t in soup.select("h1,h2,h3,p"))
        metas = []
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"): metas.append(md["content"])
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"): metas.append(og["content"])
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        combined = " ".join([title] + metas + [body_txt]).strip()
        st.caption(f"Fetched ~{len(combined)} chars from website")
        return combined[:20000]
    except Exception as e:
        st.info(f"Site fetch failed: {e}. You can paste the About text manually.")
        return ""

def enrich_company_context(base_url: str, about_text: str) -> str:
    """Pull from multiple high-signal sources, dedupe, return compact text."""
    max_pages = int(os.getenv("ENRICH_MAX_PAGES", "6"))
    seeds = []
    if base_url:
        base = normalize_url(base_url)
        seeds += [base, base + "/about", base + "/about-us", base + "/company", base + "/who-we-are"]
    # fetch with caps
    texts, used_sources, snips = [], [], {}
    for u in (seeds)[:max_pages]:
        try:
            t = fetch_text_any(u)
            if t and len(t) > 200:
                texts.append(t); used_sources.append(u); snips[u] = t
        except Exception:
            continue
    if about_text and len(about_text) > 60:
        texts.insert(0, about_text)
    combined = _dedupe_shrink(texts, max_chars=20000)
    st.session_state.sources_used = used_sources
    st.session_state.source_snips = {u: (snips.get(u) or "")[:220] for u in used_sources}
    return combined

def guess_company_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        for sub in ("www.", "m.", "app."):
            if host.startswith(sub): host = host[len(sub):]
        parts = host.split(".")
        sld = parts[-2] if len(parts) >= 2 else parts[0]
        return sld.replace("-", " ").replace("_", " ").title()
    except Exception:
        return ""

def fetch_extra_pages(base_url: str) -> str:
    if not base_url:
        return ""
    paths = ["/about", "/about-us", "/company", "/company-info", "/who-we-are"]
    combined = []
    for pth in paths:
        try:
            url2 = base_url.rstrip("/") + pth
            txt = fetch_site_text(url2)
            if txt and len(txt) > 200:
                combined.append(txt)
        except Exception:
            pass
    return "\n\n".join(combined)

def _fmt(v):
    if isinstance(v, dict):
        return "\n".join([f"- **{k}:** {v[k]}" for k in v]) or "—"
    if isinstance(v, (list, tuple)):
        return "\n".join([f"- {x}" for x in v]) or "—"
    return v if (isinstance(v, str) and v.strip()) else "—"

# =======================
# Visualization & explainers
# =======================
# --- replace your current function with this ---
def supply_chain_dot(f: CaseFacts) -> str:
    from graphviz import Digraph  # local import to avoid any import timing weirdness
    g = Digraph("supply_chain", format="svg")
    g.attr(rankdir="LR", bgcolor="transparent")

    # high-contrast for dark themes
    g.attr('node', shape="box", style="rounded", color="white", fontcolor="white", penwidth="1.4")
    g.attr('edge', color="white", fontcolor="white", penwidth="1.4", arrowsize="0.8")
    g.attr('graph', color="white")

    e1 = f.full_legal_name_1 or "Entity 1"
    e2 = f.full_legal_name_2 or "Entity 2"
    g.node("E1", e1)
    g.node("E2", e2)
    flow = (f.transaction_flows or "Goods/Services/IP").replace("\n"," ")
    g.edge("E1", "E2", label=flow)

    return g.source





def model_email_style_explainer(tx_type: str) -> str:
    t = (tx_type or "").lower()
    if "resale" in t or "lrd" in t:
        return (
            "**How LRD works:** The local entity resells finished goods, performs sales/marketing, "
            "and bears limited risks. It earns a routine operating margin (e.g., TNMM).\n\n"
            "**Example:** Parent sells goods at transfer price; Distributor sells to third-parties, "
            "keeping an arm’s-length margin (benchmarked).\n\n"
            "**Pros:** Simple to monitor; widely accepted; aligns routine returns with routine functions."
        )
    if "services" in t or "cost plus" in t:
        return (
            "**How Cost-Plus works:** Service provider charges costs plus a routine markup.\n\n"
            "**Example:** Shared services center bills OpCos at cost + X%.\n\n"
            "**Pros:** Transparent; easy to evidence; low controversy risk."
        )
    if "licensing" in t or "royalty" in t:
        return (
            "**How Royalty works:** Operator pays a % of revenue or fixed fee to the IP owner.\n\n"
            "**Example:** Brandco licenses trademark to OpCo at x% of net sales.\n\n"
            "**Pros:** Separates IP return; scalable across markets."
        )
    if "contract manufacturing" in t:
        return (
            "**How Contract Manufacturing works:** Manufacturer performs production with stripped risks, "
            "earning a routine margin.\n\n**Pros:** Clear allocation of inventory/market risks."
        )
    if "commissionaire" in t or "agency" in t:
        return (
            "**How Commissionaire works:** Agent sells on behalf of principal; revenue recognized by principal; "
            "agent earns a commission.\n\n**Pros:** Local footprint without full distributor risks."
        )
    return "**Model:** Selected based on functions, assets, and risks provided."

# =======================
# LLM helpers (classification + dispute)
# =======================
class LLM:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            st.warning("OPENAI_API_KEY missing in .env")
        self.client = OpenAI(api_key=key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = float(os.getenv("OPENAI_TEMP", "0.1"))
        self.top_p = float(os.getenv("OPENAI_TOP_P", "1"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))

    def json(self, system: str, user: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt}

TRANSACTION_TYPES = [
    "Ecommerce Resale (LRD)",
    "Wholesale Resale (LRD)",
    "Services (Cost Plus)",
    "IP Licensing (Royalty)",
    "Contract Manufacturing",
    "Commissionaire / Agency",
    "Financing / Intercompany Loans",
]

CLASSIFY_SYSTEM = (
    "You are a senior transfer pricing advisor. Classify the intercompany transaction into exactly one of the allowed labels."
)

def classify_transaction(llm: LLM, facts: dict) -> dict:
    allowed = ", ".join(TRANSACTION_TYPES)
    user = (
        "Facts as JSON (from extractor):\n"
        + json.dumps(facts, ensure_ascii=False)
        + f"\n\nReturn JSON with fields: transaction_type (one of [{allowed}]), "
          "confidence (0-1), short_rationale (string)."
    )
    return llm.json(CLASSIFY_SYSTEM, user)

ERAN_ASSISTANT_SYSTEM = (
    "You are Eran, a senior transfer pricing advisor. Be concise, practical, and clear. "
    "The user is disputing the recommended model/summary. "
    "Decide among: (1) 'update' (provide JSON with minimal field edits to facts), "
    "(2) 'escalate' (too complex; ask to speak with Eran), or (3) 'disagree'. "
    "Return JSON with keys: action ('update'|'escalate'|'disagree'), "
    "message (short explanation to the client), and optional edits (object of field->newvalue)."
)

def process_dispute_turn(llm: LLM, user_msg: str, facts: CaseFacts) -> dict:
    payload = {"current_facts": asdict(facts), "user_feedback": user_msg}
    return llm.json(ERAN_ASSISTANT_SYSTEM, json.dumps(payload, ensure_ascii=False))

# =======================
# Simple markdown templates
# =======================
IC_TMPL = """
# Intercompany Agreement (Draft)

**Parties:** {{ parties }}

**Effective Date:** {{ effective_date }}

**Transaction Type:** {{ tx_type }}

**Scope**
{{ scope }}

**Pricing**
{{ pricing }}

**FAR Summary**
{{ far }}

**Risk Allocation**
{{ risks }}

**Compliance Notes**
{{ compliance }}

---
*(Draft follows prior Incodox format; finalize after client confirmation.)*
"""

MEMO_TMPL = """
# Guidance Memo (Client-Facing)

**Company:** {{ company_name }}

**Objective**
{{ objective }}

**Recommended Model:** {{ model_name }}

**Key Assumptions**
{{ assumptions }}

**Next Steps**
- Confirm roles and facts
- Provide org chart & financials
- Approve pricing method/margins
"""

TPR_TMPL = """
# Transfer Pricing Report (Draft)

**Company:** {{ company_name }}

**Executive Summary**
{{ exec_summary }}

**Business Overview**
{{ business_overview }}

**Functional Analysis**
{{ far }}

**Method Selection & Rationale**
{{ method_selection }}

**Benchmark Indicators (Illustrative)**
{{ benchmarks }}

**Conclusion**
{{ conclusion }}

"""
# =======================
# Eran Email Summary (exact style)
# =======================
ERAN_EMAIL_TMPL = """
{% set is_lrd = 'resale' in tx_type.lower() or 'lrd' in tx_type.lower() %}
{% set is_cp  = 'cost plus' in tx_type.lower() or 'services' in tx_type.lower() %}

# {{ title or "Summary" }}

**Suggested Transfer Pricing Model:** {{ tx_type }}

## Main facts
- Parent / HQ entity: {{ f.full_legal_name_1 or "—" }}
- US/Foreign operating entity: {{ f.full_legal_name_2 or "—" }}
- Addresses: {{ (f.entity1_address or "—") }} / {{ (f.entity2_address or "—") }}
- Countries of incorporation: {{ ", ".join(f.countries_of_incorp) if f.countries_of_incorp else "—" }}
- Employees (E1/E2): {{ f.entity1_employees or 0 }} / {{ f.entity2_employees or 0 }}
- Roles & functions: {{ f.role or "—" }}
- IP / know-how: {{ f.ip_assets or "—" }}
- Current channels / flows: {{ f.transaction_flows or "—" }}

{% if is_lrd %}
## What is the LRD and how it works
Target the distributor ({{ f.full_legal_name_2 or "the US entity" }}) with a fixed **operating profit margin** (profit before interest and tax, as % of sales).  
The **intercompany transaction** (cost of purchased goods/solutions from {{ f.full_legal_name_1 or "HQ" }}) is a **plug number** that yields, at year-end, the predetermined margin that is benchmarked to comparable independent companies.  
If {{ f.full_legal_name_2 or "the distributor" }} falls within the benchmark range, the price is **arm’s length** (compliance met).

### Invoicing
Transfer pricing is assessed on an **annual** basis (aligned to the tax return). Operationally, invoice like you would a very **large client** (significant discounts) and make **periodic intercompany adjustments** (monthly/quarterly) so the distributor’s operating margin aligns to the determined margin.

### Example of how the LRD works

| {{ f.full_legal_name_2 or "Inc" }} | $ | Notes |
|---|---:|---|
| Sales | 1,000 | Sales to local clients |
| Cost of Goods | (???) | Intercompany “plug” so OP margin hits the fixed % |
| Operating Expenses | (300) | Admin, salaries, rent |
| **Operating Profit (PBIT)** | **30** | If fixed margin is **3%**, OP = 1,000 × 3% = 30 → COGS = 1,000 − 300 − 30 = **670**. If {{ f.full_legal_name_1 or "HQ" }} invoiced only 150 during the year, issue an intercompany **adjustment** for **520**. |

### Pros of the LRD model
- In growth periods, the distributor keeps a **low, fixed** profit — more profit is allocated to {{ f.full_legal_name_1 or "HQ" }}.  
- **No WHT** on cross-border **business** transactions in many jurisdictions vs. dividends/interest.  
- Aligns with facts: distributor has **no HQ/IP/R&D** and limited risks.  
- **Common** and simple to execute/monitor with **quarterly** true-ups.

{% elif is_cp %}
## Proposed Transfer Pricing approach – what should be the arm’s-length fee?
Characterize {{ f.full_legal_name_2 or "the service entity" }} as a **low-risk service provider**, compensated on a **Cost-Plus** basis.

### Costs & markup
- Costs typically include direct items (e.g., transaction fees, local accounting, regulatory fees); **exclude** items not incurred (no employees/rent if not present).  
- Typical markup range: **5%–10%** (final % to be set with benchmarks/precedents).

### Documentation
- **Intercompany agreement** supporting the Cost-Plus policy.  
- **Guidance Memo** (policy) now; full **transfer pricing report** is retrospective (for prior FY) and can be prepared later if/when required.
{% else %}
## Method overview
Based on the classification and facts, apply the standard model documentation and invoicing mechanics used in Eran’s email summaries (LRD or Cost-Plus), including annual testing and periodic true-ups.
{% endif %}

## Documents to prepare
- **Intercompany agreement** — aligns with the selected model.
- **Guidance Memo** — policy explaining methodology and controls. 
"""


def render_eran_email_summary(facts: CaseFacts, tx_type: str, title: str = "Summary Email") -> str:
    tmpl = Template(ERAN_EMAIL_TMPL)
    return tmpl.render(f=facts, tx_type=tx_type or "TBD", title=title)



def _render_docs(deliverables: list[str], ctx: dict) -> list[tuple[str, bytes]]:
    files = []
    today = dt.date.today().isoformat()
    if "Intercompany Agreement" in deliverables:
        files.append((f"IC_Agreement_{today}.md", Template(IC_TMPL).render(**ctx).encode("utf-8")))
    if "Guidance Memo" in deliverables:
        files.append((f"Guidance_Memo_{today}.md", Template(MEMO_TMPL).render(**ctx).encode("utf-8")))
    if "Transfer Pricing Report" in deliverables:
        files.append((f"TP_Report_{today}.md", Template(TPR_TMPL).render(**ctx).encode("utf-8")))
    return files

# =======================
# Session + Inputs
# =======================
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "edited" not in st.session_state:
    st.session_state.edited = {}

st.markdown("**Paste the company’s ‘About Us’ (required).** The website URL is optional and used only to enrich.")
about_text = st.text_area("About Us (required):", "", height=200)
raw_url = st.text_input("Company website URL (optional, for enrichment):", "")
deep_enrich = st.toggle(
    "Deep enrichment (multi-source)",
    value=True,
    help="About page + internal pages (deduped)."
)

# =======================
# Extract Button
# =======================
extract_disabled = len((about_text or "").strip()) < 40
if st.button("Extract company info", type="primary", disabled=extract_disabled):
    with st.spinner("Extracting…"):
        url = normalize_url(raw_url)
        source_text = about_text.strip()

        if deep_enrich and url:
            with st.spinner("Enriching context from multiple sources…"):
                source_text = enrich_company_context(url, source_text)
        elif url and len(source_text) < 1500:
            extra = fetch_extra_pages(url)
            if extra:
                source_text = (source_text + "\n\n" + extra).strip()

        st.caption(f"Context length used: {len(source_text)} chars")

        try:
            extracted_obj = extract_from_chunk(source_text[:20000], url=url)
            st.session_state.extracted = extracted_obj.model_dump()
            st.success("Extracted structured info from company profile.")
            if url:
                guess = guess_company_from_url(url)
                if guess and not st.session_state.extracted.get("full_legal_name_1"):
                    st.session_state.extracted["full_legal_name_1"] = guess
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.session_state.extracted = {}

# =======================
# Review & Edit
# =======================
extracted = st.session_state.extracted

def _as_comma_list(v):
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return v if isinstance(v, str) else ""

if extracted:
    st.markdown(
        """
        <div style="
            border:1px solid #1f2b25;
            background: linear-gradient(135deg, #0c1110 0%, #111a17 100%);
            padding:16px 18px;
            border-radius:12px;
            margin:12px 0 22px 0;
            box-shadow: 0 0 18px rgba(0,0,0,0.35);
        ">
            <div style="
                font-weight:600;
                color:#3bff95;
                font-size:15px;
                display:flex;
                align-items:center;
                gap:8px;
            ">
                <span style="font-size:18px;">✔</span>
                Review extracted fields — edit anything before generating advice
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("review_edit_form"):
        colA, colB = st.columns(2)
        with colA:
            full_legal_name_1 = st.text_input(
                "Entity 1 full legal name (Home)",
                value=st.session_state.edited.get("full_legal_name_1", extracted.get("full_legal_name_1","")),
            )
            entity1_address = st.text_area(
                "Entity 1 registered address",
                value=st.session_state.edited.get("entity1_address", extracted.get("entity1_address","")),
                height=70,
            )
            entity1_employees = st.number_input(
                "Entity 1 employees (approx.)",
                min_value=0, max_value=100000, value=int(st.session_state.edited.get("entity1_employees", extracted.get("entity1_employees") or 0)), step=1,
            )
            entity1_roles_txt = st.text_input(
                "Entity 1 roles (comma-separated)",
                value=", ".join(st.session_state.edited.get("entity1_roles", extracted.get("entity1_roles", [])))
            )
            countries_of_incorp = st.text_input(
                "Countries of incorporation (comma-separated)",
                value=st.session_state.edited.get("countries_of_incorp", _as_comma_list(extracted.get("countries_of_incorp",""))),
            )

        with colB:
            full_legal_name_2 = st.text_input(
                "Entity 2 full legal name (Foreign/Related)",
                value=st.session_state.edited.get("full_legal_name_2", extracted.get("full_legal_name_2","")),
            )
            entity2_address = st.text_area(
                "Entity 2 registered address",
                value=st.session_state.edited.get("entity2_address", extracted.get("entity2_address","")),
                height=70,
            )
            entity2_employees = st.number_input(
                "Entity 2 employees (approx.)",
                min_value=0, max_value=100000, value=int(st.session_state.edited.get("entity2_employees", extracted.get("entity2_employees") or 0)), step=1,
            )
            entity2_roles_txt = st.text_input(
                "Entity 2 roles (comma-separated)",
                value=", ".join(st.session_state.edited.get("entity2_roles", extracted.get("entity2_roles", [])))
            )

        business_model = st.text_area(
            "Business model (what the company does; markets; channels)",
            value=st.session_state.edited.get("business_model", _fmt(extracted.get("business_model",""))),
            height=110,
        )
        role = st.text_area(
            "FAR / roles (headline functions, assets, risks by entity)",
            value=st.session_state.edited.get("role", _fmt(extracted.get("role",""))),
            height=110,
        )
        ip_assets = st.text_area(
            "IP assets (brands, tech, data, know-how)",
            value=st.session_state.edited.get("ip_assets", _fmt(extracted.get("ip_assets",""))),
            height=80,
        )
        transaction_flows = st.text_area(
            "Transaction flows / pricing mechanics (who invoiced what to whom)",
            value=st.session_state.edited.get("transaction_flows", _fmt(extracted.get("transaction_flows",""))),
            height=110,
        )

        options = ["TBD"] + TRANSACTION_TYPES

        current_val = st.session_state.edited.get(
            "recommended_model",
            extracted.get("recommended_model", "TBD")
        )

        if current_val not in options:
            current_val = "TBD"

        recommended_model = st.selectbox(
            "Recommended model",
            options=options,
            index=safe_index(current_val, options, default=0),
        )

        confidence = st.slider("Confidence", 0.0, 1.0, float(st.session_state.edited.get("confidence", extracted.get("confidence") or 0.7)), 0.05)

        left, mid, right = st.columns([1,1,2])
        with left:
            submitted = st.form_submit_button("Save Edits")
        with mid:
            reset_btn = st.form_submit_button("Reset to Extracted")
        with right:
            st.caption("Saved edits are used in the summary & deliverables.")

    if submitted:
        st.session_state.edited = {
            "full_legal_name_1": full_legal_name_1.strip(),
            "full_legal_name_2": full_legal_name_2.strip(),
            "entity1_address": entity1_address.strip(),
            "entity2_address": entity2_address.strip(),
            "countries_of_incorp": [c.strip() for c in countries_of_incorp.split(",") if c.strip()],
            "entity1_employees": int(entity1_employees) if entity1_employees is not None else None,
            "entity2_employees": int(entity2_employees) if entity2_employees is not None else None,
            "business_model": business_model.strip(),
            "role": role.strip(),
            "ip_assets": ip_assets.strip(),
            "transaction_flows": transaction_flows.strip(),
            "recommended_model": recommended_model,
            "confidence": float(confidence),
            "entity1_roles": [x.strip() for x in entity1_roles_txt.split(",") if x.strip()],
            "entity2_roles": [x.strip() for x in entity2_roles_txt.split(",") if x.strip()],
        }
        st.success("Edits saved.")
        st.session_state.summary_agree = False

    if "reset_btn" in locals() and reset_btn:
        st.session_state.dispute_mode = False
        st.session_state.summary_agree = False
        st.session_state.edited = {}
        st.info("Edits cleared — reverted to extracted values.")

# ---------- formatting + dispute helpers ----------
def format_key_facts_md(f: CaseFacts) -> str:
    bullets = []
    if f.full_legal_name_1: bullets.append(f"- **Entity 1 (Home):** {f.full_legal_name_1}")
    if f.entity1_address: bullets.append(f"- **Entity 1 address:** {f.entity1_address}")
    if f.entity1_employees is not None: bullets.append(f"- **Entity 1 employees:** {f.entity1_employees}")
    if f.full_legal_name_2: bullets.append(f"- **Entity 2 (Foreign):** {f.full_legal_name_2}")
    if f.entity2_address: bullets.append(f"- **Entity 2 address:** {f.entity2_address}")
    if f.entity2_employees is not None: bullets.append(f"- **Entity 2 employees:** {f.entity2_employees}")
    if f.entity1_roles: bullets.append(f"- **Roles — Entity 1:** {', '.join(f.entity1_roles)}")
    if f.entity2_roles: bullets.append(f"- **Roles — Entity 2:** {', '.join(f.entity2_roles)}")
    if f.transaction_flows: bullets.append(f"- **Channels / Flows:** {f.transaction_flows}")
    if f.ip_assets: bullets.append(f"- **IP (candidate):** {f.ip_assets}")
    if f.countries_of_incorp: bullets.append(f"- **Country(ies):** {', '.join(f.countries_of_incorp)}")
    if not bullets:
        bullets.append("- No additional facts provided yet.")
    return "\n".join(bullets)

# =======================
# Summary for approval
# =======================
if extracted:
    st.subheader("Summary for approval")

    # Merge extracted + any edited values into one set of facts
    facts = _merge_facts(st.session_state.extracted, st.session_state.edited)

    # Classify on merged facts to get recommendation + rationale
    llm = LLM()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY in .env — cannot classify. Add the key and rerun.")
        st.stop()
    cls_summary = classify_transaction(llm, asdict(facts))

    # Supply chain diagram
    st.markdown("#### Visualize supply chain (confirm)")
    try:
        g = supply_chain_dot(facts)
        st.graphviz_chart(g)   # g is a DOT string now

    except Exception:
        st.info("Supply chain diagram unavailable — fill entity names and flows.")
    st.checkbox("I confirm the supply chain diagram is correct.", key="confirm_diagram")

    # Email-style summary (Eran exact format)
    tx_type_summary = cls_summary.get("transaction_type") or facts.recommended_model or "TBD"
    email_md = render_eran_email_summary(facts, tx_type_summary, title="Summary")
    st.markdown("### Summary of Approval (Eran email format)")
    st.markdown(email_md)
    with st.expander("Copy raw text for email"):
        st.code(email_md, language="markdown")
    # quick export for copy/paste to email
    with st.expander("Copy plain text for email (no Markdown)"):
        st.text(email_md)
    st.download_button(
        "Download Summary_Email.md",
        data=email_md.encode("utf-8"),
        file_name="Summary_Email.md",
        mime="text/markdown",
    )


    # Key facts
    st.markdown("#### Main facts used")
    st.markdown(format_key_facts_md(facts))

    # Agree / Disagree with working chat
    col_agree, col_disagree = st.columns([1,1])
    with col_agree:
        st.session_state.summary_agree = st.checkbox(
            "I confirm this summary is correct.",
            value=st.session_state.get("summary_agree", False),
        )
    with col_disagree:
        if st.button("I do not agree with this summary"):
            st.session_state.dispute_mode = True

    # Chat mode if disputed
    if st.session_state.get("dispute_mode"):
        st.divider()
        st.subheader("Discuss with advisor (chat)")
        if "chat" not in st.session_state:
            st.session_state.chat = []

        for role, msg in st.session_state.chat:
            st.chat_message(role).markdown(msg)

        user_msg = st.chat_input("Tell us what’s off — we’ll adjust or escalate to Eran.")
        if user_msg:
            st.session_state.chat.append(("user", user_msg))
            st.chat_message("user").markdown(user_msg)

            with st.chat_message("assistant"):
                resp = process_dispute_turn(llm, user_msg, facts)
                st.session_state.chat.append(("assistant", resp.get("message","")))
                st.markdown(resp.get("message",""))

                action = resp.get("action","")
                if action == "update" and isinstance(resp.get("edits"), dict):
                    st.session_state.edited.update(resp["edits"])
                    st.success("Updated facts based on your notes. Refreshing the summary…")
                    st.session_state.summary_agree = False
                    st.session_state.dispute_mode = False
                    st.rerun()
                elif action == "escalate":
                    st.info("This looks nuanced — please speak with Eran.")
                else:
                    st.info("Noted. If you still have concerns, please speak with Eran.")

# =======================
# Deliverables
# =======================
def choose_deliverables(_: None = None) -> list[str]:
    # Per Eran: all users are first-time implementations; no full long report.
    return ["Intercompany Agreement", "Guidance Memo"]

if extracted:
    st.subheader("Deliverables")
    can_generate = bool(st.session_state.get("summary_agree", False))

    if st.button("Generate advisory", type="secondary", disabled=not can_generate):
        llm = LLM()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Missing OPENAI_API_KEY in .env — cannot classify. Add the key and rerun.")
            st.stop()
        with st.spinner("Classifying transaction & choosing deliverables…"):
            facts_for_llm = {**st.session_state.extracted, **st.session_state.edited}
            cls = classify_transaction(llm, facts_for_llm)
            tx_type = cls.get("transaction_type") or facts_for_llm.get("recommended_model") or "TBD"
            deliverables = choose_deliverables()

        st.success(f"Transaction: {tx_type} — Deliverables: {', '.join(deliverables)}")
        st.json(cls)

        company_name = st.session_state.edited.get("full_legal_name_1") or extracted.get("full_legal_name_1") or "ClientCo"
        scope = st.session_state.edited.get("business_model") or extracted.get("business_model") or "Define scope of goods/services/IP."
        pricing = st.session_state.edited.get("transaction_flows") or extracted.get("transaction_flows") or "Cost Plus X% / Resale Margin / Royalty % (to confirm)."
        far = st.session_state.edited.get("role") or extracted.get("role") or "Headline FAR: key functions, assets, risks."

        ctx = {
            "parties": f"{company_name} and Related Party",
            "effective_date": dt.date.today().isoformat(),
            "tx_type": tx_type,
            "scope": scope if isinstance(scope, str) else _fmt(scope),
            "pricing": pricing if isinstance(pricing, str) else _fmt(pricing),
            "far": far if isinstance(far, str) else _fmt(far),
            "risks": "Operational, market, credit risks allocated per model.",
            "compliance": "Align with OECD; local/master file as applicable.",
            "company_name": company_name,
            "objective": "Set routine return / defendable pricing.",
            "model_name": tx_type,
            "assumptions": "Assumes routine distributor/service provider; confirm roles & FAR.",
            "exec_summary": f"Based on extracted facts, {company_name} appears to engage in {tx_type}.",
            "business_overview": scope,
            "method_selection": "Method chosen per classification (LRD → TNMM; Services → Cost Plus; Licensing → CUP/Residual).",
            "benchmarks": "Attach benchmarks from your dataset when available.",
            "conclusion": "Pricing model yields arm's-length outcome under stated assumptions.",
        }

        files = _render_docs(deliverables, ctx)

        # ----- Advisory Summary (client-facing) -----
        facts = _merge_facts(st.session_state.extracted, st.session_state.edited)
        reason = cls.get("short_rationale") or "Based on provided facts."
        req_docs = [
            "- Intercompany Agreement: outlines scope, pricing method, and responsibilities.",
            "- Guidance Memo: policy note explaining methodology and controls.",
        ]

        doc_md = f"""# Advisory Summary

**Client:** {facts.full_legal_name_1 or "ClientCo"}  
**Counterparty:** {facts.full_legal_name_2 or "RelatedCo"}  

## Suggested Transfer Pricing Model
**{tx_type}**

## Company Snapshot (confirm)
- **Entity 1 address:** {facts.entity1_address or "—"}
- **Entity 2 address:** {facts.entity2_address or "—"}
- **Employees (E1/E2):** {(facts.entity1_employees or 0)} / {(facts.entity2_employees or 0)}
- **Primary flows:** {facts.transaction_flows or "—"}
- **IP noted:** {facts.ip_assets or "—"}
- **Countries:** {(", ".join(facts.countries_of_incorp)) if facts.countries_of_incorp else "—"}

## Strategic Recommendation Summary
{reason}

### Pros of this model
{os.linesep.join(["- "+p for p in model_email_style_explainer(tx_type).split("**Pros:**")[-1].strip().splitlines() if p and not p.startswith("**")])}

### Required documents
{os.linesep.join(req_docs)}

## Main facts (from your inputs)
{format_key_facts_md(facts)}

## What I still need
- Org chart, entity registrations & addresses (confirmed)
- Employees, roles by entity (confirmed)
- Financials / margins by function
- Contracts (if any)
- Benchmarks (we will attach)

## Next Steps
1. Confirm this summary (or flag anything that looks off).
2. Provide the items listed under “What I still need.”
3. We will produce the intercompany agreement and guidance memo for review.
"""

        st.download_button(
            "Download Advisory_Summary.md",
            data=doc_md.encode("utf-8"),
            file_name="Advisory_Summary.md",
            mime="text/markdown",
        )

        # ----- Save case log -----
        import pathlib, datetime as _dt
        case_dir = pathlib.Path("knowledge/cases")
        case_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        case_payload = {
            "facts": asdict(facts),
            "classification": cls,
            "advisory_md": doc_md,
        }
        case_path = case_dir / f"CASE_{ts}.json"
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case_payload, f, ensure_ascii=False, indent=2)
        st.caption(f"Saved case log → {case_path}")

        st.subheader("Downloads")
        for name, data in files:
            if name.startswith("TP_Report_"):
                # Not offering TPR by default per Eran; keep generated file available if needed.
                continue
            st.download_button(label=f"Download {name}", data=data, file_name=name, mime="text/markdown")

# =======================
# NOTE: Removed deprecated <12 months> branching. Deliverables fixed.
# =======================

# --------------------- requirements.txt hint ---------------------
# streamlit>=1.36
# openai>=1.51
# requests>=2.32
# beautifulsoup4>=4.12
# jinja2>=3.1
# python-dotenv>=1.0
# graphviz>=0.20
