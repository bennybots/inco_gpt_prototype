from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Extracted(BaseModel):
    full_legal_name_1: Optional[str] = None
    full_legal_name_2: Optional[str] = None
    countries_of_incorp: List[str] = []
    business_model: Optional[Literal[
        "ecomm_resale","wholesale","services","manufacturing","platform","marketplace","other"
    ]] = None
    role: Optional[Literal[
        "LRD","distributor","agent","service_provider","principal","other"
    ]] = None
    founded_year: Optional[int] = None
    age_lt_12mo: Optional[bool] = None
    ip_assets: List[str] = []
    transaction_flows: Optional[str] = None
    recommended_model: Optional[Literal[
        "LRD","CostPlus","Commissionaire","Principal","TBD"
    ]] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    @field_validator("confidence", mode="before")
    def normalize_confidence(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v == "high":
                return 0.9
            elif v == "medium":
                return 0.6
            elif v == "low":
                return 0.3
        return v



SYS = (
    "You are a precise JSON extractor. "
    "Return ONLY valid JSON that matches the requested schema. "
    "If a field is unknown, use null or an empty array."
)

USER_TMPL = """
Extract the following fields and return ONLY valid JSON matching EXACTLY this schema:

{{
  "full_legal_name_1": string or null,
  "full_legal_name_2": string or null,
  "countries_of_incorp": array of strings,
  "business_model": one of ["ecomm_resale","wholesale","services","manufacturing","platform","marketplace","other"],
  "role": one of ["LRD","distributor","agent","service_provider","principal","other"],
  "founded_year": number or null,
  "age_lt_12mo": boolean or null,
  "ip_assets": array of strings,
  "transaction_flows": string or null,
  "recommended_model": one of ["LRD","CostPlus","Commissionaire","Principal","TBD"],
  "confidence": "high" | "medium" | "low"
}}

TEXT:
{chunk}
"""


import json
from openai import OpenAI

import json

def extract_from_chunk(chunk: str, url: str | None = None) -> Extracted:
    if not chunk.strip():
        return Extracted()

    prompt = f"""
You are an information extraction assistant.

When possible, infer facts from context and the URL. If you are confident, fill the field;
if not, return null (or empty array). Prefer short, clean values.

Website hint (may help for name/country/role): {url or "unknown"}

Return ONLY valid JSON that matches EXACTLY this schema:
{{
  "full_legal_name_1": string or null,
  "full_legal_name_2": string or null,
  "countries_of_incorp": array of strings,
  "business_model": one of ["ecomm_resale","wholesale","services","manufacturing","platform","marketplace","other"],
  "role": one of ["LRD","distributor","agent","service_provider","principal","other"],
  "founded_year": number or null,
  "age_lt_12mo": boolean or null,
  "ip_assets": array of strings,
  "transaction_flows": string or null,
  "recommended_model": one of ["LRD","CostPlus","Commissionaire","Principal","TBD"],
  "confidence": "high" | "medium" | "low"
}}

TEXT:
{chunk}
"""


    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a precise JSON extractor. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip()

    # Debug
    print("DEBUG CHUNK LENGTH:", len(chunk))
    print("DEBUG GPT RAW:", raw)

    # ---------- helpers (inside function, outside try) ----------
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [str(x)]

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            v = x.strip().lower()
            if v in ("true", "yes", "y", "1"):
                return True
            if v in ("false", "no", "n", "0"):
                return False
        return None
    # -----------------------------------------------------------

    try:
        data = json.loads(raw)

        # Normalize a few fields defensively
        data["countries_of_incorp"] = _as_list(data.get("countries_of_incorp"))
        data["ip_assets"] = _as_list(data.get("ip_assets"))
        data["age_lt_12mo"] = _to_bool(data.get("age_lt_12mo"))

        return Extracted(**data)

    except Exception as e:
        print("JSON parse error:", e)
        return Extracted()
