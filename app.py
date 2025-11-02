# app.py — Incodox GPT (fits your current structure)
# Drop-in replacement: keeps your extractor + adds classification, deliverables, downloads.

# --- imports & env ---
import os, json, glob, requests, datetime as dt
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import streamlit as st

# your existing extractor (unchanged)
from app.ie.extractors import extract_from_chunk

from dataclasses import dataclass, asdict, field
from typing import Optional, List

@dataclass
class CaseFacts:
    full_legal_name_1: str = ""
    full_legal_name_2: str = ""
    countries_of_incorp: List[str] = field(default_factory=list)
    founded_year: Optional[int] = None
    age_lt_12mo: Optional[bool] = None
    business_model: str = ""
    role: str = ""
    ip_assets: str = ""
    transaction_flows: str = ""
    recommended_model: str = "TBD"
    confidence: Optional[float] = None
    entity1_roles: List[str] = field(default_factory=list)
    entity2_roles: List[str] = field(default_factory=list)
    ip_name_suggestion: List[str] = field(default_factory=list)


def _merge_facts(extracted: dict, edited: dict) -> CaseFacts:
    d = {**(extracted or {}), **(edited or {})}
    # normalize lists
    if isinstance(d.get("countries_of_incorp"), str):
        d["countries_of_incorp"] = [x.strip() for x in d["countries_of_incorp"].split(",") if x.strip()]
    return CaseFacts(
        full_legal_name_1=d.get("full_legal_name_1",""),
        full_legal_name_2=d.get("full_legal_name_2",""),
        countries_of_incorp=d.get("countries_of_incorp") or [],
        founded_year=d.get("founded_year"),
        age_lt_12mo=d.get("age_lt_12mo"),
        business_model=d.get("business_model",""),
        role=d.get("role",""),
        ip_assets=d.get("ip_assets",""),
        transaction_flows=d.get("transaction_flows",""),
        recommended_model=d.get("recommended_model","TBD"),
        confidence=d.get("confidence"),
        entity1_roles=d.get("entity1_roles") or [],
        entity2_roles=d.get("entity2_roles") or [],
        ip_name_suggestion=d.get("ip_name_suggestion") or [],
    )


print("DEBUG — Loaded key:", os.getenv("OPENAI_API_KEY"))
DEBUG = os.getenv("RETRIEVER_DEBUG", "0") == "1"

# --- streamlit config must be first ---
st.set_page_config(page_title="Incodox GPT - Prototype", page_icon="🧠", layout="wide")
st.title("Incodox GPT")

# =======================
# Helpers
# =======================
import re, time, random
from urllib.parse import urlencode

# --- smart fetch with UA rotate + text-proxy fallback (robust) ---
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

def fetch_text_any(url: str, timeout: int = None) -> str:
    if not url: return ""
    timeout = timeout or int(os.getenv("ENRICH_TIMEOUT", "12"))
    # try direct with rotated UAs
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
    # fallback: text proxy (works across many JS/403 sites)
    try:
        p = urlparse(url)
        proxy_url = f"https://r.jina.ai/{p.scheme}://{p.netloc}{p.path or '/'}"
        r2 = requests.get(proxy_url, timeout=timeout)
        if r2.status_code == 200 and r2.text:
            return r2.text.strip()
    except Exception:
        pass
    return ""

# --- simple Google search via SerpAPI (optional) ---
def search_web_queries(queries: list[str], n: int = 5) -> list[str]:
    provider = os.getenv("SEARCH_PROVIDER", "none").lower()
    key = os.getenv("SERPAPI_KEY", "")
    urls: list[str] = []
    if provider == "serpapi" and key:
        for q in queries:
            try:
                params = {"engine": "google", "q": q, "num": n, "api_key": key}
                r = requests.get("https://serpapi.com/search", params=params, timeout=10)
                data = r.json()
                for item in (data.get("organic_results") or [])[:n]:
                    u = item.get("link")
                    if u and u not in urls:
                        urls.append(u)
            except Exception:
                continue
    return urls

# --- enrichment orchestration ---
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

def enrich_company_context(base_url: str, about_text: str) -> str:
    """Pull from multiple high-signal sources, dedupe, return compact text."""
    max_pages = int(os.getenv("ENRICH_MAX_PAGES", "6"))
    seeds = []

    if base_url:
        base = normalize_url(base_url)
        seeds += [base, base + "/about", base + "/about-us", base + "/company", base + "/who-we-are"]

    # quick heuristics for company handle
    domain_name = guess_company_from_url(base_url or "")
    qname = domain_name or ""

    # optional search (SerpAPI)
    extra = []
    if qname:
        extra += search_web_queries([
            f"{qname} about",
            f"{qname} company profile",
            f"{qname} intercompany",
            f"{qname} business model",
        ], n=5)

    # LinkedIn & Wikipedia guesses
    if qname:
        extra += [
            f"https://en.wikipedia.org/wiki/{qname.replace(' ', '_')}",
            f"https://www.linkedin.com/company/{qname.replace(' ', '-')}/about/",
        ]

    # fetch with caps
    texts = []
    used_sources: list[str] = []
    snips: dict[str, str] = {}
    for u in (seeds + extra)[:max_pages]:
        try:
            t = fetch_text_any(u)
            if t and len(t) > 200:
                texts.append(t)
                used_sources.append(u)
                snips[u] = t
        except Exception:
            continue

    # include pasted About (highest signal)
    if about_text and len(about_text) > 60:
        texts.insert(0, about_text)

    combined = _dedupe_shrink(texts, max_chars=20000)

    # Save sources + tiny preview (first 220 chars) for UI
    st.session_state.sources_used = used_sources  # list[str]
    st.session_state.source_snips = {u: (snips.get(u) or "")[:220] for u in used_sources}
    return combined



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

        body_txt = " ".join(
            t.get_text(" ", strip=True) for t in soup.select("h1,h2,h3,p")
        )

        metas = []
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            metas.append(md["content"])
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            metas.append(og["content"])

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        combined = " ".join([title] + metas + [body_txt]).strip()
        length = len(combined) if combined else 0

        st.caption(f"Fetched ~{length} chars from website")
        return combined[:20000]

    except Exception as e:
        st.info(f"Site fetch failed: {e}. You can paste the About text manually.")
        return ""


def _fmt(v):
    if isinstance(v, dict):
        return "\n".join([f"- **{k}:** {v[k]}" for k in v]) or "—"
    if isinstance(v, (list, tuple)):
        return "\n".join([f"- {x}" for x in v]) or "—"
    return v if (isinstance(v, str) and v.strip()) else "—"

# ---------- Helpers for clearer summary / Eran-style reasoning ----------
def _bool_to_yesno(v: bool | None) -> str:
    return "Yes" if v else ("No" if v is not None else "Unknown")

def older_than_12mo_flag(f: CaseFacts) -> bool | None:
    """Infer 'older than 12 months' from founded_year and/or age_lt_12mo."""
    try:
        yr = f.founded_year
        if f.age_lt_12mo is True:
            return False
        if f.age_lt_12mo is False:
            return True
        if yr:
            return (dt.date.today().year - int(yr)) >= 2  # conservative: >=2 calendar years → >12m
    except Exception:
        pass
    return None  # unknown

def deliverable_reason(f: CaseFacts) -> str:
    flag = older_than_12mo_flag(f)
    if f.age_lt_12mo is True:
        return ("**Startup package recommended** because the entity is **younger than 12 months**. "
                "Prepare an Intercompany Agreement and a Guidance Memo now; produce a Transfer Pricing Report after first fiscal year.")
    if f.age_lt_12mo is False:
        return ("**Transfer Pricing Report recommended** because the entity appears **older than 12 months** "
                "and should have sufficient financial data for a Local File.")
    if flag is True:
        return ("**Transfer Pricing Report recommended** inferred from the founded year (older than ~12 months).")
    if flag is False:
        return ("**Startup package recommended** inferred from a very recent founded year (<~12 months).")
    return ("Insufficient age info. Defaulting to **Startup package (IC Agreement + Guidance Memo)** until age is confirmed.")

def pros_for_model(tx_type: str) -> list[str]:
    tx_type = (tx_type or "").lower()
    if "resale" in tx_type or "lrd" in tx_type:
        return [
            "Aligns with a sales/marketing-focused entity that bears limited risks.",
            "Common OECD-accepted distributor framework with clear routine margin.",
            "Simplifies compliance and monitoring for intercompany transactions.",
            "Helps reduce overall tax friction on cross-border flows.",
        ]
    if "services" in tx_type or "cost plus" in tx_type:
        return [
            "Clear routine return on provider’s operating costs.",
            "Straightforward to document and benchmark for support functions.",
            "Lower controversy risk vs. entrepreneurial returns.",
        ]
    if "licensing" in tx_type or "royalty" in tx_type:
        return [
            "Separates returns to IP owner from operating entities.",
            "Supports scalability across multiple markets and users.",
            "Flexible to align with CUP/benchmark ranges when available.",
        ]
    if "contract manufacturing" in tx_type:
        return [
            "Risk-stripped manufacturer earns a routine return.",
            "Clear allocation of inventory/market risks away from manufacturer.",
        ]
    if "commissionaire" in tx_type or "agency" in tx_type:
        return [
            "Sales presence without full-risk distributor exposure.",
            "Remuneration tied to commissions vs. gross margins.",
        ]
    if "financing" in tx_type or "loans" in tx_type:
        return [
            "Allows arm’s-length pricing of intercompany funding.",
            "Separates financial risk/return from operating entities.",
        ]
    return ["Model selected based on functions, assets, and risks described."]

def required_docs_for(deliverables: list[str]) -> list[str]:
    out = []
    if "Intercompany Agreement" in deliverables:
        out.append("Intercompany Agreement: outlines scope, pricing method, and responsibilities.")
    if "Guidance Memo" in deliverables:
        out.append("Guidance Memo: policy note explaining methodology and controls.")
    if "Transfer Pricing Report" in deliverables:
        out.append("Transfer Pricing Report (Local File): OECD-aligned documentation for the fiscal year.")
    return out


def extract_source_text(raw_url: str, about_text: str) -> tuple[str, str]:
    url = normalize_url(raw_url)
    site_text = fetch_site_text(url) if url else ""
    source_text = (about_text or site_text).strip()
    return source_text, url


def guess_company_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        for sub in ("www.", "m.", "app."):
            if host.startswith(sub):
                host = host[len(sub):]
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



# =======================
# Minimal LLM helpers (classification)
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



def choose_deliverables(age_lt_12mo: bool | None) -> list[str]:
    if age_lt_12mo is None:
        return ["Intercompany Agreement", "Guidance Memo"]
    return ["Intercompany Agreement", "Guidance Memo"] if age_lt_12mo else ["Transfer Pricing Report"]


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

from jinja2 import Template

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
# 🟡 Session + Input Fields
# =======================
if "extracted" not in st.session_state:
    st.session_state.extracted = {}

raw_url = st.text_input("Enter the company website URL (optional):", "")
about_text = st.text_area(
    "Paste the company's 'About Us' or general description here (optional):",
    ""
)
# Toggle enrichment mode
deep_enrich = st.toggle(
    "Deep enrichment (multi-source)",
    value=True,
    help="About page + internal pages + Google/LinkedIn/Wikipedia (deduped)"
)

# =======================
# 🟢 Extract Button
# =======================
if st.button("Extract company info", type="primary"):
    with st.spinner("Extracting…"):
        # Build base params
        url = normalize_url(raw_url)
        source_text = (about_text or "").strip()

        if deep_enrich:
            with st.spinner("Enriching context from multiple sources…"):
                source_text = enrich_company_context(url, source_text)
        else:
            # fallback to light (homepage + about pages)
            if url and len(source_text) < 1500:
                extra = fetch_extra_pages(url)
                if extra:
                    source_text = (source_text + "\n\n" + extra).strip()

        st.caption(f"Context length used: {len(source_text)} chars")

        try:
            # ✅ Run extractor in BOTH modes
            extracted = extract_from_chunk(source_text[:20000], url=url)
            st.session_state.extracted = extracted.model_dump()
            st.success("Extracted structured info from company profile.")

            # backfill domain guess if empty
            if url:
                guess = guess_company_from_url(url)
                if guess and not st.session_state.extracted.get("full_legal_name_1"):
                    st.session_state.extracted["full_legal_name_1"] = guess

        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.session_state.extracted = {}


# =======================
# 🟠 Review & Edit (client can override any field)
# =======================
extracted = st.session_state.extracted

# Utility: prefer edited values if available
if "edited" not in st.session_state:
    st.session_state.edited = {}

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
                "Company Name 1",
                value=st.session_state.edited.get("full_legal_name_1", extracted.get("full_legal_name_1", "")),
            )
            full_legal_name_2 = st.text_input(
                "Company Name 2",
                value=st.session_state.edited.get("full_legal_name_2", extracted.get("full_legal_name_2", "")),
            )
            countries_of_incorp = st.text_input(
                "Countries (comma-separated)",
                value=st.session_state.edited.get(
                    "countries_of_incorp",
                    _as_comma_list(extracted.get("countries_of_incorp", "")),
                ),
            )
            founded_year = st.number_input(
                "Founded Year (optional)",
                min_value=1800,
                max_value=2100,
                value=int(extracted.get("founded_year") or 2000),
                step=1,
            )
            age_lt_12mo = st.checkbox(
                "< 12 months old?",
                value=bool(st.session_state.edited.get("age_lt_12mo", extracted.get("age_lt_12mo", False))),
            )
        with colB:
            business_model = st.text_area(
                "Business Model",
                value=st.session_state.edited.get("business_model", _fmt(extracted.get("business_model", ""))),
                height=120,
            )
            role = st.text_area(
                "Role (headline FAR/roles)",
                value=st.session_state.edited.get("role", _fmt(extracted.get("role", ""))),
                height=120,
            )
            ip_assets = st.text_area(
                "IP Assets",
                value=st.session_state.edited.get("ip_assets", _fmt(extracted.get("ip_assets", ""))),
                height=90,
            )
            transaction_flows = st.text_area(
                "Transaction Flows / Pricing",
                value=st.session_state.edited.get("transaction_flows", _fmt(extracted.get("transaction_flows", ""))),
                height=120,
            )

        rec_default = extracted.get("recommended_model") or "TBD"
        options = ["TBD"] + TRANSACTION_TYPES
        recommended_model = st.selectbox(
            "Recommended Model",
            options=options,
            index=options.index(st.session_state.edited.get("recommended_model", rec_default if rec_default in options else "TBD")),
        )
        confidence = st.slider(
            "Confidence (if overriding, adjust)",
            min_value=0.0, max_value=1.0, value=float(extracted.get("confidence") or 0.7), step=0.05,
        )

        left, mid, right = st.columns([1,1,2])
        with left:
            submitted = st.form_submit_button("Save Edits")
        with mid:
            reset_btn = st.form_submit_button("Reset to Extracted")
        with right:
            st.caption("Saved edits are used below when generating the advisory.")

    if submitted:
        st.session_state.edited = {
            "full_legal_name_1": full_legal_name_1.strip(),
            "full_legal_name_2": full_legal_name_2.strip(),
            "countries_of_incorp": [c.strip() for c in countries_of_incorp.split(",") if c.strip()],
            "founded_year": int(founded_year) if founded_year else None,
            "age_lt_12mo": bool(age_lt_12mo),
            "business_model": business_model.strip(),
            "role": role.strip(),
            "ip_assets": ip_assets.strip(),
            "transaction_flows": transaction_flows.strip(),
            "recommended_model": recommended_model,
            "confidence": float(confidence),
        }
        st.success("Edits saved. You can now generate the advisory.")

    if reset_btn:
        st.session_state.edited = {}
        st.info("Edits cleared — reverted to extracted values.")

# ---------- NEW: readable key-facts formatting + dispute chat helpers ----------
def format_key_facts_md(f: CaseFacts) -> str:
    bullets = []
    if f.full_legal_name_1: bullets.append(f"- **Entity 1 (Home):** {f.full_legal_name_1}")
    if f.full_legal_name_2: bullets.append(f"- **Entity 2 (Foreign):** {f.full_legal_name_2}")
    if f.entity1_roles: bullets.append(f"- **Roles — Entity 1:** {', '.join(f.entity1_roles)}")
    if f.entity2_roles: bullets.append(f"- **Roles — Entity 2:** {', '.join(f.entity2_roles)}")
    if f.transaction_flows: bullets.append(f"- **Channels / Flows:** {f.transaction_flows}")
    if f.ip_assets: bullets.append(f"- **IP (candidate):** {f.ip_assets}")
    if f.countries_of_incorp: bullets.append(f"- **Country(ies):** {', '.join(f.countries_of_incorp)}")
    if f.founded_year: bullets.append(f"- **Founded:** {f.founded_year}")
    if f.age_lt_12mo is not None: bullets.append(f"- **Age < 12 months?:** {bool(f.age_lt_12mo)}")
    if not bullets:
        bullets.append("- No additional facts provided yet.")
    return "\n".join(bullets)

ERAN_ASSISTANT_SYSTEM = (
    "You are Eran, a senior transfer pricing advisor. Be concise, practical, and clear. "
    "The user is disputing the recommended model/summary. "
    "Decide among: (1) 'update' (provide JSON with minimal field edits to facts), "
    "(2) 'escalate' (too complex; ask to speak with Eran), or (3) 'disagree'. "
    "Return JSON with keys: action ('update'|'escalate'|'disagree'), "
    "message (short explanation to the client), and optional edits (object of field->newvalue)."
)

def process_dispute_turn(llm: LLM, user_msg: str, facts: CaseFacts) -> dict:
    payload = {
        "current_facts": asdict(facts),
        "user_feedback": user_msg
    }
    return llm.json(ERAN_ASSISTANT_SYSTEM, json.dumps(payload, ensure_ascii=False))

# ===== Step D: Summary for approval =====
if extracted:
    st.subheader("Summary for approval")

    # Merge extracted + any edited values into one set of facts
    facts = _merge_facts(st.session_state.extracted, st.session_state.edited)

    # Gentle nudge if age flag conflicts with founded year
    try:
        if facts.age_lt_12mo and facts.founded_year and facts.founded_year < dt.date.today().year - 1:
            st.info("⚠️ You marked **< 12 months** but the founded year looks older. Double-check this.")
    except Exception:
        pass

    # Classify on merged facts to get the current best recommendation + rationale
  # Classify on merged facts to get recommendation + rationale
llm = LLM()
cls_summary = classify_transaction(llm, asdict(facts))
tx_type_summary = cls_summary.get("transaction_type") or facts.recommended_model or "TBD"
rationale_summary = cls_summary.get("short_rationale") or "Based on the provided facts and roles."

st.markdown(f"### Suggested transfer pricing model\n**{tx_type_summary}**")

# Company snapshot (bold, confirmable)
age_known = older_than_12mo_flag(facts)
st.markdown("#### Company snapshot (confirm)")
st.markdown(
    "\n".join([
        f"- **Entity 1:** {facts.full_legal_name_1 or '—'}",
        f"- **Entity 2:** {facts.full_legal_name_2 or '—'}",
        f"- **Founded year:** {facts.founded_year or '—'}",
        f"- **Older than 12 months:** {_bool_to_yesno(age_known)}",
        f"- **Primary channel / flows:** {facts.transaction_flows or '—'}",
        f"- **IP noted:** {facts.ip_assets or '—'}",
        f"- **Countries:** {', '.join(facts.countries_of_incorp) or '—'}",
    ])
)

# Why this fits
st.markdown("#### Why this fits")
st.write(rationale_summary)

# Why this deliverable (reasoning)
age_reason = deliverable_reason(facts)
st.markdown("#### Why this deliverable")
st.write(age_reason)

# Key facts (still readable bullets)
st.markdown("#### Main facts used")
st.markdown(format_key_facts_md(facts))

# Dispute / agreement controls (unchanged)
col_agree, col_disagree = st.columns([1,1])
with col_agree:
    st.session_state.summary_agree = st.checkbox(
        "I confirm this summary is correct.",
        value=st.session_state.get("summary_agree", False)
    )
with col_disagree:
    if st.button("I do not agree with this summary"):
        st.session_state.dispute_mode = True


    # Dispute flow
    col_agree, col_disagree = st.columns([1,1])
    with col_agree:
        st.session_state.summary_agree = st.checkbox(
            "I confirm this summary is correct.",
            value=st.session_state.get("summary_agree", False)
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
                    # merge edits into edited and refresh summary on next rerun
                    st.session_state.edited.update(resp["edits"])
                    st.success("Thanks — I’ve updated the facts and refreshed the summary.")
                    st.session_state.summary_agree = False
                    st.session_state.dispute_mode = False
                    st.rerun()
                elif action == "escalate":
                    st.info("Please speak with Eran about this specific situation.")
                else:  # disagree or unknown
                    st.info("I disagree based on the provided information. If you still have concerns, please talk to Eran.")

# =======================
# 🔵 Classify + Generate Deliverables (uses edits if present)
# =======================
if extracted:
    st.subheader("Deliverables")

    facts_for_llm = {**extracted, **st.session_state.edited}

    can_generate = bool(st.session_state.get("summary_agree", False))
    if st.button("Generate advisory", type="secondary", disabled=not can_generate):
        llm = LLM()
        with st.spinner("Classifying transaction & choosing deliverables…"):
            cls = classify_transaction(llm, facts_for_llm)
            tx_type = cls.get("transaction_type") or facts_for_llm.get("recommended_model") or "TBD"
            age_flag = facts_for_llm.get("age_lt_12mo")
            deliverables = choose_deliverables(age_flag)

        st.success(f"Transaction: {tx_type} — Deliverables: {', '.join(deliverables)}")
        st.json(cls)

        company_name = facts_for_llm.get("full_legal_name_1") or "ClientCo"
        scope = facts_for_llm.get("business_model") or "Define scope of goods/services/IP."
        pricing = facts_for_llm.get("transaction_flows") or "Cost Plus X% / Resale Margin / Royalty % (to confirm)."
        far = facts_for_llm.get("role") or "Headline FAR: key functions, assets, risks."

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

        # ----- Single Advisory Summary (client-facing) -----
        facts = _merge_facts(st.session_state.extracted, st.session_state.edited)
        reason = cls.get("short_rationale") or "Based on provided facts."
        deliverable = "IC Agreement + Guidance Memo" if facts.age_lt_12mo else "Transfer Pricing Report (summary)"

        pros_list = pros_for_model(tx_type)
req_docs = required_docs_for(deliverables)

doc_md = f"""# Advisory Summary

**Client:** {facts.full_legal_name_1 or "ClientCo"}  
**Counterparty:** {facts.full_legal_name_2 or "RelatedCo"}  

## Suggested Transfer Pricing Model
**{tx_type}**

## Company Snapshot (confirm)
- **Founded year:** {facts.founded_year or "—"}
- **Older than 12 months:** {_bool_to_yesno(older_than_12mo_flag(facts))}
- **Primary channel / flows:** {facts.transaction_flows or "—"}
- **IP noted:** {facts.ip_assets or "—"}
- **Countries:** {(", ".join(facts.countries_of_incorp)) if facts.countries_of_incorp else "—"}

## Strategic Recommendation Summary
{reason}

### Why this deliverable
{deliverable_reason(facts)}

### Pros of this model
{os.linesep.join(f"- {p}" for p in pros_list)}

### Required documents
{os.linesep.join(f"- {d}" for d in req_docs)}

## Main facts (from your inputs)
{format_key_facts_md(facts)}

## What I still need
- Org chart & entity registrations
- Financials / margins by function
- Contracts (if any)
- Benchmarks (we will attach)

## Next Steps
1. Confirm this summary (or flag anything that looks off).
2. Provide the items listed under “What I still need.”
3. We will produce the {deliverable.lower()} for your review.
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
            st.download_button(label=f"Download {name}", data=data, file_name=name, mime="text/markdown")

# =======================
# 🧹 NOTE: Removed/disabled old retriever/Chroma code that referenced undefined vars
# (MAX_CHARS, OVERLAP, client, EMBED_MODEL, chroma, col, DOC_IDS, DOCS_DIR, USE_GOOGLE).
# This file now runs without those dependencies. Add back retrieval later if needed.

# --------------------- requirements.txt hint ---------------------
# streamlit>=1.36
# openai>=1.51
# requests>=2.32
# beautifulsoup4>=4.12
# jinja2>=3.1
# python-dotenv>=1.0
