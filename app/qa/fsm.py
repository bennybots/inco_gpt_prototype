from typing import Dict, List

MANDATORY = ["full_legal_name_1", "full_legal_name_2"]

PROMPTS = {
  "full_legal_name_1": "What is the full legal name of the first company?",
  "full_legal_name_2": "What is the full legal name of the second company?",
  "age_lt_12mo": "Was the company incorporated within the last 12 months? (yes/no)",
  "role": ("Which option best describes your operating role? "
           "(E-commerce reseller, Wholesale distributor, Service provider, "
           "Agent/Commissionaire, Principal/Manufacturer, Platform/Marketplace)")
}

def missing_keys(extracted: Dict) -> List[str]:
    need = []
    for k in MANDATORY:
        if not extracted.get(k):
            need.append(k)
    if extracted.get("age_lt_12mo") is None:
        need.append("age_lt_12mo")
    if not extracted.get("role"):
        need.append("role")
    return need

def next_questions(extracted: Dict) -> List[Dict]:
    keys = missing_keys(extracted)
    return [{"id": k, "question": PROMPTS[k]} for k in keys]
