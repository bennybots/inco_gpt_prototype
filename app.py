from __future__ import annotations

# --- Drop-in replacement: form-first UX + preview-on-click + deliverable picker ---

# --- imports & env ---
import os, json, requests, datetime as dt, re, random, pathlib
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
    entity1_address: str = ""
    entity2_address: str = ""

    # Jurisdictions
    countries_of_incorp: List[str] = field(default_factory=list)

    # People / roles
    entity1_employees: Optional[int] = None
    entity2_employees: Optional[int] = None
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
    max_pages = int(os.getenv("ENRICH_MAX_PAGES", "6"))
    seeds = []
    if base_url:
        base = normalize_url(base_url)
        seeds += [base, base + "/about", base + "/about-us", base + "/company", base + "/who-we-are"]
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
# Diagram edit assistant
# =======================
DIAGRAM_EDIT_SYSTEM = (
    "You are a structured editor for a simple 2-box supply-chain diagram.\n"
    "Given the current facts and the user's note, output JSON ONLY with any of:\n"
    "{"
    "  'entity1_name': <str>,"
    "  'entity2_name': <str>,"
    "  'flow_label': <str>,"
    "  'direction': 'E1->E2'|'E2->E1'"
    "}\n"
    "Keep changes minimal. Do not invent entities beyond two. If unclear, return {}."
)

# =======================
# LLM helpers
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

def propose_diagram_update(llm: LLM, user_msg: str, facts: CaseFacts) -> dict:
    payload = {"current_facts": asdict(facts), "instruction": user_msg}
    return llm.json(DIAGRAM_EDIT_SYSTEM, json.dumps(payload, ensure_ascii=False))

def apply_diagram_edits(facts: CaseFacts, edits: dict) -> CaseFacts:
    f = CaseFacts(**asdict(facts))
    if edits.get("entity1_name"):
        f.full_legal_name_1 = edits["entity1_name"].strip()
    if edits.get("entity2_name"):
        f.full_legal_name_2 = edits["entity2_name"].strip()
    if edits.get("flow_label"):
        f.transaction_flows = edits["flow_label"].strip()
    if edits.get("direction") == "E2->E1":
        f.transaction_flows = (f.transaction_flows or "Goods/Services/IP") + "  (E2 → E1)"
    return f

# =======================
# Visualization & explainers
# =======================
def supply_chain_dot(f: CaseFacts) -> str:
    from graphviz import Digraph  # local import
    g = Digraph("supply_chain", format="svg")
    g.attr(rankdir="LR", bgcolor="transparent")
    g.attr('node', shape="box", style="rounded", color="white", fontcolor="white", penwidth="1.4")
    g.attr('edge', color="white", fontcolor="white", penwidth="1.4", arrowsize="0.8")
    g.attr('graph', color="white")
    e1 = f.full_legal_name_1 or "Entity 1"
    e2 = f.full_legal_name_2 or "Entity 2"
    g.node("E1", e1); g.node("E2", e2)
    flow = (f.transaction_flows or "Goods/Services/IP").replace("\n"," ")
    g.edge("E1", "E2", label=flow)
    return g.source

TRANSACTION_TYPES = [
    "Ecommerce Resale (LRD)",
    "Wholesale Resale (LRD)",
    "Services (Cost Plus)",
    "IP Licensing (Royalty)",
    "Contract Manufacturing",
    "Commissionaire / Agency",
    "Financing / Intercompany Loans",
]

CLASSIFY_SYSTEM = "You are a senior transfer pricing advisor. Classify the intercompany transaction into exactly one of the allowed labels."

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
# Deliverables logic (Eran-style + LLM check)
# =======================
def _bool(v) -> bool:
    return bool(v) and str(v).strip().lower() not in {"0", "none", "n/a", "na", "-"}

def _num(v, default=0):
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return default

DELIV_RULES_DOC = """
You are a transfer pricing assistant channeling Eran's email-summary pattern.
Choose deliverables based on these ideas:

- Intercompany Agreement (IC): forward-looking implementation of a TP model for ongoing dealings (LRD resale, Cost-Plus services, Commissionaire, Contract Mfg). Especially for first-time setups. Often paired with a short Guidance Memo.
- Guidance Memo: concise policy write-up explaining the model, pricing mechanics, true-up logic, responsibilities, and controls. Used for quick onboarding or when full Local/Master File not (yet) required.
- Transfer Pricing Report (TPR): retrospective documentation to defend prior or current fiscal year, meet Local/Master File requirements, or handle audit/controversy; more likely when materiality is high, multi-country footprint, IP licensing/royalty, financing/loans, or historical exposure exists.

Signals:
- If facts imply new/first-time implementation → IC + Guidance Memo.
- If there’s historical activity (already trading last FY), audit flags, multi-jurisdictions, royalties, or financing → include TPR.
- If confidence is low or facts are thin → include Guidance Memo even if IC is selected, and propose discovery.
- If transaction is purely policy (shared services with small footprint) → Guidance Memo alone can be enough, but IC is still common if there’s a recurring related-party payment.
"""

def choose_deliverables_smart(facts: CaseFacts, cls: dict, llm: LLM | None = None) -> tuple[list[str], str]:
    """
    Returns (deliverables, rationale_text).
    Heuristics first, then optional LLM nudge. Never removes IC+Memo if clearly indicated.
    """
    tx = (cls.get("transaction_type") or facts.recommended_model or "").lower()
    conf_raw = cls.get("confidence")
    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.7

    multi_country = len(facts.countries_of_incorp) >= 2
    has_ip = bool(facts.ip_assets)
    employees_total = (facts.entity1_employees or 0) + (facts.entity2_employees or 0)
    flows = (facts.transaction_flows or "").lower()
    roles = (facts.role or "").lower()

    looks_new_setup = (not flows or "tbd" in flows) and employees_total <= 25
    likely_historical = any(k in flows for k in ["invoice", "sold", "true-up", "prior", "fy20"])
    risky_types = any(k in tx for k in ["licensing", "royalty", "financing", "loan"])
    clear_operational = any(k in tx for k in ["resale", "lrd", "services", "commissionaire", "contract manufacturing"])

    deliverables = set()
    rationale = []

    if clear_operational:
        deliverables |= {"Intercompany Agreement", "Guidance Memo"}
        rationale.append("Operational related-party dealings → IC + Memo for policy clarity.")

    if looks_new_setup and clear_operational:
        rationale.append("Likely first-time implementation → IC + Memo instead of full TPR.")

    if multi_country or risky_types or has_ip or likely_historical:
        deliverables.add("Transfer Pricing Report")
        rationale.append("Multi-country/IP/financing or historical exposure → add TPR for defensibility.")

    if conf < 0.55:
        deliverables.add("Guidance Memo")
        rationale.append("Low model confidence → include Memo to document assumptions.")

    if not deliverables:
        deliverables = {"Intercompany Agreement", "Guidance Memo"}
        rationale.append("Default fallback → IC + Memo.")

    if llm:
        llm_payload = {
            "facts": asdict(facts),
            "classification": cls,
            "current_choice": sorted(deliverables),
        }
        try:
            llm_out = llm.json(DELIV_RULES_DOC, json.dumps(llm_payload, ensure_ascii=False))
            llm_choice = llm_out.get("deliverables") or llm_out.get("choice") or []
            if isinstance(llm_choice, list):
                merged = set(deliverables) | set(llm_choice)
                if clear_operational:
                    merged |= {"Intercompany Agreement", "Guidance Memo"}
                deliverables = merged
                if llm_out.get("rationale"):
                    rationale.append(f"LLM check: {llm_out['rationale']}")
        except Exception:
            pass

    return sorted(deliverables), " ".join(rationale) or "Based on operational profile and risk signals."

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
# Session
# =======================
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "edited" not in st.session_state:
    st.session_state.edited = {}
if "show_preview" not in st.session_state:
    st.session_state.show_preview = False
if "classification_cache" not in st.session_state:
    st.session_state.classification_cache = None

# =======================
# Inputs (About + URL)
# =======================
st.markdown("**Paste the company’s ‘About Us’ (required).** The website URL is optional and used only to enrich.")
about_text = st.text_area("About Us (required):", "", height=200)
raw_url = st.text_input("Company website URL (optional, for enrichment):", "")
deep_enrich = st.toggle("Deep enrichment (multi-source)", value=True, help="About page + internal pages (deduped).")

# =======================
# Extract Button (no preview yet)
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
            # hide preview until user clicks the preview button
            st.session_state.show_preview = False
            st.session_state.classification_cache = None
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.session_state.extracted = {}
            st.session_state.show_preview = False
            st.session_state.classification_cache = None

# =======================
# Review & Edit (FORM ONLY)
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
            padding:16px 18px; border-radius:12px; margin:12px 0 22px 0;
            box-shadow: 0 0 18px rgba(0,0,0,0.35);
        ">
            <div style="font-weight:600;color:#3bff95;font-size:15px;display:flex;align-items:center;gap:8px;">
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
                min_value=0, max_value=100000,
                value=int(st.session_state.edited.get("entity1_employees", extracted.get("entity1_employees") or 0)), step=1,
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
                min_value=0, max_value=100000,
                value=int(st.session_state.edited.get("entity2_employees", extracted.get("entity2_employees") or 0)), step=1,
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
        current_val = st.session_state.edited.get("recommended_model", extracted.get("recommended_model", "TBD"))
        if current_val not in options:
            current_val = "TBD"
        recommended_model = st.selectbox("Recommended model", options=options, index=safe_index(current_val, options, default=0))
        confidence = st.slider("Confidence", 0.0, 1.0, float(st.session_state.edited.get("confidence", extracted.get("confidence") or 0.7)), 0.05)

        left, mid, right = st.columns([1,1,2])
        with left:
            submitted = st.form_submit_button("Save Edits")
        with mid:
            reset_btn = st.form_submit_button("Reset to Extracted")
        with right:
            generate_preview_clicked = st.form_submit_button("Generate advisory preview", type="primary")

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
        # hide preview until refreshed
        st.session_state.show_preview = False
        st.session_state.classification_cache = None

    if "reset_btn" in locals() and reset_btn:
        st.session_state.dispute_mode = False
        st.session_state.summary_agree = False
        st.session_state.edited = {}
        st.session_state.show_preview = False
        st.session_state.classification_cache = None
        st.info("Edits cleared — reverted to extracted values.")

    if "generate_preview_clicked" in locals() and generate_preview_clicked:
        st.session_state.show_preview = True
        st.session_state.summary_agree = False   # must reconfirm on each preview

# ---------- formatting helper ----------
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
# PREVIEW (only after click)
# =======================
if extracted and st.session_state.show_preview:
    st.subheader("Advisory preview")

    # Merge facts
    facts = _merge_facts(st.session_state.extracted, st.session_state.edited)

    # Classify only when previewing (cache per preview)
    if st.session_state.classification_cache is None:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Missing OPENAI_API_KEY in .env — cannot classify. Add the key and rerun.")
            st.stop()
        llm = LLM()
        st.session_state.classification_cache = classify_transaction(llm, asdict(facts))
    cls = st.session_state.classification_cache
    tx_type = cls.get("transaction_type") or facts.recommended_model or "TBD"
    # Smart deliverables prediction (Eran-style)
    pred_deliverables, deliv_rationale = choose_deliverables_smart(facts, cls, llm=LLM())
    st.markdown("#### Predicted deliverables")
    st.write(", ".join(pred_deliverables))
    st.caption(deliv_rationale)

    # Supply chain diagram
    st.markdown("#### Visualize supply chain (confirm)")
    try:
        g_dot = supply_chain_dot(facts)
        st.graphviz_chart(g_dot, use_container_width=True)
    except Exception:
        st.info("Supply chain diagram unavailable — fill entity names and flows.")

    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.checkbox("I confirm the supply chain diagram is correct.", key="confirm_diagram")
    with cols[1]:
        open_edit = st.button("This diagram is wrong", key="diagram_wrong_btn")

    # mini chat to fix diagram
    if open_edit or st.session_state.get("diagram_edit_open"):
        st.session_state.diagram_edit_open = True
        st.divider()
        st.subheader("Adjust the diagram")
        if "diagram_chat" not in st.session_state:
            st.session_state.diagram_chat = []
        for role, msg in st.session_state.diagram_chat[-6:]:
            st.chat_message(role).markdown(msg)
        user_msg = st.chat_input("Describe what to change (names, flow text, or direction).", key="diagram_chat_input")
        if user_msg:
            st.session_state.diagram_chat.append(("user", user_msg))
            with st.chat_message("assistant"):
                llm = LLM()
                edits = propose_diagram_update(llm, user_msg, facts) or {}
                preview = ("Proposed edits:\n" + "\n".join(f"- **{k}** → {v}" for k, v in edits.items())) if edits else "I couldn't infer a clear change. Please be more specific."
                st.markdown(preview)
                if edits:
                    new_facts = apply_diagram_edits(facts, edits)
                    st.session_state.edited.update({
                        "full_legal_name_1": new_facts.full_legal_name_1,
                        "full_legal_name_2": new_facts.full_legal_name_2,
                        "transaction_flows": new_facts.transaction_flows,
                    })
                    st.success("Updated. Re-rendering diagram…")
                    st.session_state.diagram_chat.append(("assistant", preview))
                    st.session_state.confirm_diagram = False
                    # refresh classification on change
                    st.session_state.classification_cache = None
                    st.rerun()
                else:
                    st.session_state.diagram_chat.append(("assistant", preview))

    # Main facts + summary agree/dispute
    st.markdown("#### Main facts used")
    st.markdown(format_key_facts_md(facts))

    col_agree, col_disagree = st.columns([1, 1])
    with col_agree:
        st.session_state.summary_agree = st.checkbox(
            "I confirm this summary is correct.",
            value=st.session_state.get("summary_agree", False),
        )
    with col_disagree:
        if st.button("I do not agree with this summary", key="dispute_btn"):
            st.session_state.dispute_mode = True

    if st.session_state.get("dispute_mode"):
        st.divider()
        st.subheader("Discuss with advisor (chat)")
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for role, msg in st.session_state.chat:
            st.chat_message(role).markdown(msg)
        user_msg2 = st.chat_input("Tell us what’s off — we’ll adjust or escalate to Eran.", key="dispute_chat_input")
        if user_msg2:
            st.session_state.chat.append(("user", user_msg2))
            st.chat_message("user").markdown(user_msg2)
            with st.chat_message("assistant"):
                llm = LLM()
                resp = process_dispute_turn(llm, user_msg2, facts)
                st.session_state.chat.append(("assistant", resp.get("message", "")))
                st.markdown(resp.get("message", ""))
                action = resp.get("action", "")
                if action == "update" and isinstance(resp.get("edits"), dict):
                    st.session_state.edited.update(resp["edits"])
                    st.success("Updated facts based on your notes. Refreshing the preview…")
                    st.session_state.summary_agree = False
                    st.session_state.classification_cache = None
                    st.session_state.dispute_mode = False
                    st.rerun()
                elif action == "escalate":
                    st.info("This looks nuanced — please speak with Eran.")
                else:
                    st.info("Noted. If you still have concerns, please speak with Eran.")

    # Render a quick advisory summary preview (email-style content)
    st.markdown("#### Advisory summary (email style)")
    reason = cls.get("short_rationale") or "Based on provided facts."
    st.caption(f"Classification confidence: {cls.get('confidence')}")
    st.markdown(render_eran_email_summary(facts, tx_type, title="Summary for Approval"))
    st.markdown(f"**Strategic Recommendation Summary:**\n\n{reason}")

    # =======================
    # Deliverables Picker (after preview & confirmation)
    # =======================
    st.divider()
    st.subheader("Would you like me to generate the documents?")
    st.caption("Per Eran, most first-time cases use Intercompany Agreement + Guidance Memo. A full Transfer Pricing Report is optional.")

    default_ic   = "Intercompany Agreement" in pred_deliverables
    default_memo = "Guidance Memo" in pred_deliverables
    default_tpr  = "Transfer Pricing Report" in pred_deliverables


    col1, col2, col3 = st.columns(3)
    with col1:
        want_ic = st.checkbox("Intercompany Agreement", value=default_ic)
    with col2:
        want_memo = st.checkbox("Guidance Memo", value=default_memo)
    with col3:
        want_tpr = st.checkbox("Transfer Pricing Report (optional)", value=default_tpr)

    can_generate_docs = st.session_state.get("summary_agree", False) and (want_ic or want_memo or want_tpr)
    if st.button("Generate documents", type="secondary", disabled=not can_generate_docs):
        deliverables = []
        if want_ic: deliverables.append("Intercompany Agreement")
        if want_memo: deliverables.append("Guidance Memo")
        if want_tpr: deliverables.append("Transfer Pricing Report")

        company_name = facts.full_legal_name_1 or "ClientCo"
        scope = facts.business_model or "Define scope of goods/services/IP."
        pricing = facts.transaction_flows or "Cost Plus X% / Resale Margin / Royalty % (to confirm)."
        far = facts.role or "Headline FAR: key functions, assets, risks."

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

        # lightweight client-facing advisory summary file
        advisory_md = f"""# Advisory Summary

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

### Required documents
{os.linesep.join(['- '+d for d in deliverables])}

## Main facts (from your inputs)
{format_key_facts_md(facts)}

## Next Steps
1. Confirm this summary (or flag anything that looks off).
2. Provide org chart, registrations, addresses, employees & roles, and financials.
3. We will finalize the selected document(s) above.
"""
        st.download_button(
            "Download Advisory_Summary.md",
            data=advisory_md.encode("utf-8"),
            file_name="Advisory_Summary.md",
            mime="text/markdown",
        )

        # Save case log
        case_dir = pathlib.Path("knowledge/cases")
        case_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        case_payload = {"facts": asdict(facts), "classification": cls, "advisory_md": advisory_md}
        case_path = case_dir / f"CASE_{ts}.json"
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case_payload, f, ensure_ascii=False, indent=2)
        st.caption(f"Saved case log → {case_path}")

        st.subheader("Downloads")
        for name, data in files:
            st.download_button(label=f"Download {name}", data=data, file_name=name, mime="text/markdown")
