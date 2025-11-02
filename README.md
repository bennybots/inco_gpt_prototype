# Incodox GPT — Prototype...
Smart tax logic assistant for multinational clients

## Demo
• Short video walkthrough: <ADD LINK>  
• Screenshots below

## What it does
Incodox GPT guides an intake, classifies the engagement, and prepares draft outputs:
1) Transfer Pricing Report (TPR), or  
2) Intercompany Agreement + Guidance Memo.

## Status (Aug 2025)
• Core Streamlit UI running locally  
• Classifier + prompt builder wired  
• Word templates loaded from `/templates`  
• Document-generation stubs in place  
• Cloud deploy pending account unblock; local demo available

## How it works (high level)
1. Intake: initial questions to scope the IC transactions  
2. Classification: TPR vs IC Agreement path  
3. Drafting: fills selected Word templates with structured fields (stub today)  
4. Export: prepares docs for download (coming next)

## Tech
Streamlit, Python, OpenAI API, python-docx, pandas

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
