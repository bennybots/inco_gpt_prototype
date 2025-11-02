import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

import json, requests
import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from urllib.parse import urlparse

st.set_page_config(page_title="Incodox GPT – Prototype", page_icon="🧠", layout="wide")

def extract_source_text(raw_url: str, about_text: str) -> tuple[str, str]:
    """Normalizes URL, fetches site text (if any), and prefers pasted About text."""
    url = normalize_url(raw_url)
    site_text = fetch_site_text(url) if url else ""
    source_text = (about_text or site_text).strip()
    return source_text, url



def normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    try:
        p = urlparse(raw)
        if not p.netloc:  # e.g., they pasted just "https://"
            return ""
        # drop trailing slash-only
        return raw.rstrip("/")
    except Exception:
        return ""


def safe_json_loads(s: str):
    if not s or not s.strip():
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        st.warning("Could not parse extracted info. Using empty defaults.")
        return {}

def fetch_site_text(url: str, timeout=12) -> str:
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"},
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Body text
        body_txt = " ".join(t.get_text(" ", strip=True) for t in soup.select("h1,h2,h3,p"))

        # Meta fallbacks
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

def _fmt(v):
    # Pretty-print dicts/lists as bullets, pass through strings
    if isinstance(v, dict):
        items = [f"- **{k}:** {v[k]}" for k in v]
        return "\n".join(items) if items else "—"
    if isinstance(v, (list, tuple)):
        items = [f"- {x}" for x in v]
        return "\n".join(items) if items else "—"
    return v if (isinstance(v, str) and v.strip()) else "—"

# =======================
# URL + About inputs  →  extraction (button-driven)
# =======================
if "extracted" not in st.session_state:
    st.session_state.extracted = {}

raw_url = st.text_input("Enter the company website URL (optional):", "")
about_text = st.text_area("Paste the company's 'About Us' or general description here (optional):", "")

if st.button("Extract company info", type="primary"):
    with st.spinner("Extracting…"):
        source_text, url = extract_source_text(raw_url, about_text)

        if not source_text:
            st.info("No readable website text (cookie wall/JS) and nothing pasted. You can still fill the fields below manually.")
            st.session_state.extracted = {}
        else:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.responses.create(
                    model="gpt-4.1",
                    response_format={"type": "json_object"},
                    input=[{
                        "role": "user",
                        "content": (
                            "Extract key company facts as JSON with keys: "
                            "name, country, group, business_activity, main_roles, ip_ownership, "
                            "ic_transactions_summary. Use short strings.\n\n"
                            f"SOURCE TEXT:\n{source_text}"
                        )
                    }],
                )
                st.session_state.extracted = safe_json_loads(resp.output_text)
                st.success("Extracted info from company profile.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.session_state.extracted = {}

# Use extracted values everywhere else
extracted = st.session_state.extracted

if extracted:
    st.markdown(
        """
        <div style="
            border:1px solid #e0f2e9;
            background:#f0fff6;
            padding:16px 18px;
            border-radius:14px;
            margin:8px 0 18px 0;
        ">
            <div style="font-weight:600;color:#067647;margin-bottom:6px;">
                ✓ Extracted info from company profile
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fields = [
        ("Business Activity", extracted.get("business_activity")),
        ("Main Roles",         extracted.get("main_roles")),
        ("IP Ownership",       extracted.get("ip_ownership")),
        ("Purpose",            extracted.get("purpose")),
        ("Supply Chain",       extracted.get("supply_chain")),
        ("Country",            extracted.get("country")),
        ("Group",              extracted.get("group")),
        ("Company Name",       extracted.get("name")),
    ]

    col1, col2 = st.columns(2)
    half = (len(fields) + 1) // 2
    for idx, (label, value) in enumerate(fields):
        col = col1 if idx < half else col2
        col.markdown(f"**{label}:**")
        col.markdown(_fmt(value))
        col.markdown("---")


# =======================
# Env + Client setup
# =======================
# (dotenv loaded above; no second call here)

# Accept either DOC_IDS or DOC_ID (single ID)
_doc_ids_env = os.getenv("DOC_IDS") or os.getenv("DOC_ID") or ""
DOC_IDS = [x.strip() for x in _doc_ids_env.split(",") if x.strip()]
USE_GOOGLE = len(DOC_IDS) > 0

# Accept either GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT
if not os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = os.getenv("GOOGLE_SERVICE_ACCOUNT", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"
client = OpenAI(api_key=OPENAI_API_KEY)

# Toggle verbose logs with RETRIEVER_DEBUG=1 in .env
DEBUG = os.getenv("RETRIEVER_DEBUG", "0") == "1"

# =======================
# Config
# =======================
EMBED_MODEL = "text-embedding-3-small"
DOCS_DIR = "data/email_summaries"
MAX_CHARS = 1200
OVERLAP = 200

# Persistent Chroma collection
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST", "1") == "1"
if CHROMA_PERSIST:
    os.makedirs("knowledge/chroma", exist_ok=True)
    chroma = chromadb.PersistentClient(path="./knowledge/chroma")
else:
    chroma = chromadb.Client()  # in-memory

col = chroma.get_or_create_collection(name="eran_summaries")

# =======================
# Helpers
# =======================
def _log(msg: str):
    if DEBUG:
        print(f"[retriever] {msg}")

def _chunk_text(text: str, source_title: str, source_url: str = "") -> List[Dict]:
    if not text:
        return []
    n = len(text)
    size = max(200, min(MAX_CHARS, n))
    overlap = min(OVERLAP, size - 1)
    step = max(1, size - overlap)

    chunks: List[Dict] = []
    idx = 0
    start = 0
    while start < n:
        end = min(n, start + size)
        piece = text[start:end]
        chunks.append({
            "id": f"{source_title}:{idx}",
            "text": piece,
            "meta": {"doc_title": source_title, "doc_url": source_url or "", "chunk_index": idx},
        })
        idx += 1
        if end == n:
            break
        start += step
    return chunks

def _embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# =======================
# Loaders
# =======================
def _load_google_docs() -> List[Dict]:
    """
    Load text from Google Docs listed in DOC_IDS and chunk it.
    Cache-first: read knowledge/google_cache/<doc_id>.txt if present.
    Falls back to a single cached download only if cache is missing.
    """
    cache_dir = os.getenv("GOOGLE_CACHE_DIR", "knowledge/google_cache")
    os.makedirs(cache_dir, exist_ok=True)

    out: List[Dict] = []
    for did in DOC_IDS:
        cache_path = os.path.join(cache_dir, f"{did}.txt")
        txt = ""

        # 1) Try cache
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                _log(f"Cache hit for {did} ({len(txt)} chars).")
            except Exception as e:
                _log(f"Cache read error for {did}: {e}")

        # 2) Fallback: one attempt to download + write cache
        if not (txt or "").strip():
            _log(f"Cache miss/empty for {did}; attempting cached download once.")
            from google_docs_loader import download_docs_as_text
            docs = download_docs_as_text([did], use_cache=True)
            txt = (docs.get(did) or "").strip()
            if txt:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(txt)
                    _log(f"Wrote cache for {did} ({len(txt)} chars).")
                except Exception as e:
                    _log(f"Cache write error for {did}: {e}")
            else:
                _log(f"WARNING: Google doc {did} returned empty text after fallback.")
                continue  # skip this doc

        # 3) If we have text, normalize + chunk
        url = f"https://docs.google.com/document/d/{did}/edit"
        txt = txt.lstrip("\ufeff")
        out.extend(_chunk_text(txt, source_title=f"google_doc:{did}", source_url=url))

    _log(f"Loaded {len(out)} chunks from {len(DOC_IDS)} Google Doc(s) (cache-first).")
    return out

def _load_local_txt() -> List[Dict]:
    """Load .txt files from DOCS_DIR and chunk them."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    out = []
    for fp in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        title = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read()
        out.extend(_chunk_text(txt, source_title=title, source_url=""))
    _log(f"Loaded {len(out)} chunks from local TXT files.")
    return out

# =======================
# Index sync
# =======================
def _collection_is_empty() -> bool:
    try:
        return col.count() == 0
    except Exception:
        return True

def sync_index_if_needed(force: bool = False) -> None:
    try:
        if not force and col.count() > 0:
            _log("Index already populated; skipping rebuild.")
            return
    except Exception as e:
        _log(f"col.count() failed: {e}")

    _log(f"Building index. USE_GOOGLE={USE_GOOGLE}")
    chunks = []
    try:
        chunks = _load_google_docs() if USE_GOOGLE else _load_local_txt()
    except Exception as e:
        _log(f"Loader ERROR: {e}")
        raise

    _log(f"Chunk count loaded: {len(chunks)}")
    if not chunks:
        _log("No chunks to index (check doc sharing / DOC_IDS / local files).")
        return

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]

    _log("Creating embeddings…")
    try:
        embs = _embed(docs)
    except Exception as e:
        _log(f"Embeddings ERROR: {e}")
        raise
    _log(f"Embeddings created: {len(embs)}")

    _log("Upserting into Chroma…")
    try:
        col.upsert(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
    except Exception as e:
        _log(f"Chroma upsert ERROR: {e}")
        raise
    _log("Upsert complete.")

def force_reindex() -> None:
    _log("Force reindex requested. Deleting collection…")
    try:
        chroma.delete_collection("eran_summaries")
    except Exception as e:
        _log(f"Delete collection note: {e} (safe to ignore if first run)")
    global col
    col = chroma.get_or_create_collection(name="eran_summaries")
    _log("Rebuilding index…")
    sync_index_if_needed(force=True)
    try:
        _log(f"Rebuild complete. New count: {col.count()}")
    except Exception as e:
        _log(f"col.count() after rebuild failed: {e}")

# =======================
# Retrieval
# =======================
def retrieve(query: str, k: int = 5) -> List[Dict]:
    # Ensure index exists
    sync_index_if_needed()

    if col.count() == 0:
        _log("Collection is empty; returning no hits.")
        return []

    q_emb = _embed([query])[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances", "ids"],
    )

    hits = []
    if res and res.get("documents"):
        for doc, meta, dist, _id in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["ids"][0],
        ):
            hits.append(
                {"id": _id, "text": doc, "meta": meta or {}, "distance": float(dist)}
            )
    _log(f"Retrieved {len(hits)} hits for query: {query[:80]!r}")
    return hits

def build_context(query: str, k: int = 5) -> Tuple[str, List[str]]:
    hits = retrieve(query, k)
    if not hits:
        return "", []

    context = "\n\n---\n\n".join(h["text"] for h in hits)

    def _safe(meta: dict, key: str, default: str = ""):
        return meta.get(key, default) if isinstance(meta, dict) else default

    sources = [
        f'{_safe(h["meta"], "doc_title", "unknown")} (chunk {_safe(h["meta"], "chunk_index", 0)})'
        + (f' — {_safe(h["meta"], "doc_url")}' if _safe(h["meta"], "doc_url") else "")
        for h in hits
    ]
    return context, sources
