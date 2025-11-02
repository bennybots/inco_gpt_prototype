import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# NEW: retrieval helper
try:
    from retriever import build_context
except Exception:
    build_context = None  # app still runs without the retriever

# ========== Setup ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Incodox – TP Advisory Chat", layout="wide")
st.title("TP Advisory Chat – Phase 1 Prototype")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in .env. Add it and restart the app.")

client = OpenAI(api_key=OPENAI_API_KEY)

st.write(
    "Fill in the fields below. On submit, the app generates a professional advisory "
    "summary (left) and structured JSON for downstream automation (right)."
)

# ========== Sidebar: Sync button (optional) ==========
with st.sidebar:
    st.subheader("Knowledge")
    if st.button("Sync from Google Doc"):
        import subprocess, sys
        result = subprocess.run([sys.executable, "sync_single_doc.py"], capture_output=True, text=True)
        st.text(result.stdout or "")
        if result.stderr:
            st.error(result.stderr)
    use_doc_context = st.checkbox("Use Google Doc context", value=True,
                                  help="Pull top snippets from Eran's summaries if retriever.py is available.")

# ========== Helpers ==========
def months_since(date_str: str):
    """Return months since a YYYY-MM-DD date. If invalid, return None."""
    if not date_str:
        return None
    try:
        d = datetime.fromisoformat(date_str)
    except ValueError:
        return None
    now = datetime.now()
    return (now.year - d.year) * 12 + (now.month - d.month)

def build_heuristic_notes(has_local_revenue, books_revenue_locally, entity_purpose, incorporation_date):
    """Gentle hints only; do not override facts."""
    hints = []
    age_months = months_since(incorporation_date)
    if age_months is not None:
        if age_months < 12:
            hints.append("Entity age <12 months: favor Startup Package (IC Agreement + Guidance Memo).")
        else:
            hints.append("Entity age ≥12 months: consider preparing a Local File for the prior fiscal year.")
    if (has_local_revenue or "").lower() == "yes" and (books_revenue_locally or "").lower() == "yes":
        hints.append("Direct local sales booked locally: consider LRD where facts fit.")
    elif "service" in (entity_purpose or "").lower():
        hints.append("Service support profile: consider Cost Plus if low risk and centrally directed.")
    return " ".join(hints)

SYSTEM_PROMPT = """You are a senior transfer pricing advisor. Your job is to classify the intercompany transaction and provide a strategic recommendation to a finance leader of a multinational SMB, in the same tone and style used in professional client advisory emails.

Output sections:
1) Suggested Transfer Pricing Model
2) Entity Characterization
3) Strategic Recommendation Summary (2–4 short paragraphs)
4) Pros of this Model (bulleted)
5) Required Documents
6) Next Steps
"""

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "characterization": {"type": "string"},
        "recommendation": {"type": "string"},
        "pros": {"type": "array", "items": {"type": "string"}},
        "documents_required": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["model", "characterization", "recommendation", "pros", "documents_required", "next_steps"],
    "additionalProperties": False
}

# ========== UI (guided inputs) ==========
with st.form("tp_form"):
    company_url = st.text_input("Company Website (used to infer industry)")
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.text_input("Foreign Entity Country")
    with c2:
        incorporation_date = st.date_input("Incorporation Date", format="YYYY-MM-DD")
    with c3:
        purpose = st.selectbox(
            "Purpose of the Foreign Entity",
            ["", "Hire talent", "Sell products locally", "Logistics / fulfillment", "Provide services to group"],
            index=0
        )

    c4, c5 = st.columns(2)
    with c4:
        has_local_revenue = st.selectbox("Do customers pay the foreign entity directly?", ["", "Yes", "No"], index=0)
    with c5:
        books_revenue_locally = st.selectbox("Is that revenue booked in the entity's local books?", ["", "Yes", "No"], index=0)

    ip_owner = st.text_input("Who owns the IP / brand?")

    teams_options = [
        "R&D", "Product", "Sales", "Marketing", "Administration", "Customer Success",
        "Manufacturing", "Logistics", "Packaging", "Design", "QA", "Other"
    ]
    teams = st.multiselect("Teams in the foreign entity", teams_options)

    notes = st.text_area("Anything else worth noting? (optional)")

    submitted = st.form_submit_button("Generate Advisory")

# ========== Submit Handler ==========
if submitted:
    # Basic validation
    if not company_url or not country or not incorporation_date:
        st.error("Please fill Company Website, Country, and Incorporation Date.")
        st.stop()

    # Normalize date
    incorporation_date_str = (
        incorporation_date.strftime("%Y-%m-%d")
        if hasattr(incorporation_date, "strftime")
        else str(incorporation_date)
    )

    # Build the prompt
    heuristic_notes = build_heuristic_notes(
        has_local_revenue, books_revenue_locally, purpose, incorporation_date_str
    )

    user_prompt = f"""Based on the following client information, classify the intercompany transaction and provide a clear, professional recommendation:

- Foreign entity country: {country}
- Incorporation date: {incorporation_date_str}
- Company website: {company_url}
- Purpose of the entity: {purpose}
- Customers pay foreign entity directly: {has_local_revenue}
- Revenue booked in local books: {books_revenue_locally}
- Teams in the foreign entity: {", ".join(teams) if teams else "N/A"}
- IP owned by: {ip_owner}
- Notes: {notes or "N/A"}

Heuristic hints (use only if consistent with facts): {heuristic_notes}

When relevant, use the website to infer industry context.
If the foreign entity is younger than 12 months, suggest the Startup Package (IC Agreement + Guidance Memo).
If older than 12 months, suggest preparing a Local File for the prior fiscal year.
"""

    # NEW: pull context from Google Doc (if enabled and retriever is available)
    doc_context = ""
    doc_sources = []
    if use_doc_context and build_context is not None:
        try:
            # use the composed prompt as a proxy for the query
            doc_context, doc_sources = build_context(user_prompt, k=5)
        except Exception as e:
            st.info(f"Doc context unavailable ({e}). Proceeding without it.")

    # --- Call 1: advisory email-style text ---
    with st.spinner("Generating advisory summary..."):
        try:
            # include context if present
            combined_user = (
                (f"Context from Eran's Google Doc:\n{doc_context}\n\n" if doc_context else "")
                + user_prompt
            )

            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.4,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": combined_user},
                ],
            )
            advisory_text = completion.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI error (summary): {e}")
            st.stop()

    # --- Call 2: strict JSON (schema-guided) ---
    schema_text = json.dumps(STRUCTURED_SCHEMA, indent=2)
    json_instructions = f"""Return STRICT JSON only. It MUST validate against this JSON Schema:

{schema_text}

Transform the following advisory text into the JSON structure:

--- BEGIN ADVISORY TEXT ---
{advisory_text}
--- END ADVISORY TEXT ---
"""

    with st.spinner("Extracting structured JSON..."):
        try:
            json_completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You output ONLY JSON. No prose, no markdown."},
                    {"role": "user", "content": json_instructions},
                ],
            )
            raw_json = json_completion.choices[0].message.content
            try:
                structured = json.loads(raw_json)
            except Exception:
                # Fallback: minimal structure if parsing fails
                structured = {
                    "model": "",
                    "characterization": "",
                    "recommendation": advisory_text,
                    "pros": [],
                    "documents_required": [],
                    "next_steps": [],
                }
        except Exception as e:
            st.error(f"OpenAI error (JSON): {e}")
            st.stop()

    # ========== Display Results ==========
    left, right = st.columns(2)

    with left:
        st.subheader("AI-Generated Advisory Summary")
        st.write(advisory_text)
        if doc_sources:
            with st.expander("Sources (Google Doc)"):
                for s in doc_sources:
                    st.write("- " + s)

    with right:
        st.subheader("Structured Output (JSON)")
        st.code(json.dumps(structured, indent=2), language="json")

        # Optional: download JSON
        st.download_button(
            label="Download JSON",
            data=json.dumps(structured, indent=2).encode("utf-8"),
            file_name="tp_advisory_output.json",
            mime="application/json",
        )

    st.success("Done. You can now share this screen in your meeting with Eran.")
