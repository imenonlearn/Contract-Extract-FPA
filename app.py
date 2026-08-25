"""
Contract Key Terms Extractor — quick prototype

Upload PDF contracts, define whatever fields you want extracted,
and get a table (+ Excel export) with the extracted values per document.

Run:
    pip install streamlit pdfplumber pandas openpyxl requests anthropic python-dotenv
    streamlit run app.py

API keys:
    Create a file named `.env` in the same folder as this script with lines like:
        ANTHROPIC_API_KEY=sk-ant-...
        GROQ_API_KEY=gsk_...
    Never commit this file to GitHub — keep it local only (see .gitignore).

Backends:
    - Claude (Anthropic API, default) — reads ANTHROPIC_API_KEY from .env, or enter it in the sidebar
    - Groq — reads GROQ_API_KEY from .env, or enter it in the sidebar
    - Ollama (local) — run `ollama serve` and `ollama pull llama3.2` first
    - OpenAI-compatible API (OpenAI, Azure OpenAI, etc.) — enter base URL + key in sidebar
"""

import io
import json
import os
import re

import anthropic
import pandas as pd
import pdfplumber
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Contract Key Terms Extractor", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "fields" not in st.session_state:
    st.session_state.fields = [
        {"name": "Counterparty", "hint": "The other party to the contract (not our company)"},
        {"name": "Effective Date", "hint": "Date the contract starts / becomes effective"},
        {"name": "Term / Expiry", "hint": "Contract duration, end date, or renewal terms"},
        {"name": "Contract Value", "hint": "Total or annual fee/value, with currency"},
    ]

if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------------------------------------------------------------
# Sidebar: backend config
# ---------------------------------------------------------------------------
st.sidebar.header("LLM backend")
backend = st.sidebar.radio("Choose backend", ["Claude (Anthropic API)", "Groq", "Ollama (local)", "OpenAI-compatible API"])

claude_api_key, claude_model = None, None
groq_api_key, groq_model = None, None
ollama_url, ollama_model = None, None
api_base, api_key, api_model = None, None, None

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError, st.errors.StreamlitAPIException):
        return None


if backend == "Claude (Anthropic API)":
    if os.environ.get("ANTHROPIC_API_KEY"):
        claude_api_key = os.environ["ANTHROPIC_API_KEY"]
        st.sidebar.success("Using API key from .env")
    elif get_secret("ANTHROPIC_API_KEY"):
        claude_api_key = get_secret("ANTHROPIC_API_KEY")
        st.sidebar.success("Using configured API key")
    else:
        claude_api_key = st.sidebar.text_input("Anthropic API key", type="password")
    claude_model = st.sidebar.text_input("Model", "claude-sonnet-5")
elif backend == "Groq":
    if os.environ.get("GROQ_API_KEY"):
        groq_api_key = os.environ["GROQ_API_KEY"]
        st.sidebar.success("Using API key from .env")
    elif get_secret("GROQ_API_KEY"):
        groq_api_key = get_secret("GROQ_API_KEY")
        st.sidebar.success("Using configured API key")
    else:
        groq_api_key = st.sidebar.text_input("Groq API key", type="password")
    groq_model = st.sidebar.text_input("Model", "llama-3.3-70b-versatile")
elif backend == "Ollama (local)":
    ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
    ollama_model = st.sidebar.text_input("Model", "llama3.2")
else:
    api_base = st.sidebar.text_input("Base URL", "https://api.openai.com/v1")
    api_key = st.sidebar.text_input("API key", type="password")
    api_model = st.sidebar.text_input("Model", "gpt-4o-mini")

max_chars = st.sidebar.slider(
    "Max characters of contract text sent to the model",
    2000, 40000, 15000, step=1000,
    help="Long contracts get truncated to this length to keep prompts manageable.",
)

st.sidebar.caption(
    "Prototype only — no persistence, no auth, no multi-tenancy. "
    "Everything runs locally in this session."
)


# ---------------------------------------------------------------------------
# Step 1: define fields
# ---------------------------------------------------------------------------
st.title("Contract Key Terms Extractor")
st.subheader("1. Define the fields you want extracted")

for i, field in enumerate(st.session_state.fields):
    c1, c2, c3 = st.columns([2, 5, 1])
    field["name"] = c1.text_input("Field name", field["name"], key=f"name_{i}")
    field["hint"] = c2.text_input("What to look for", field["hint"], key=f"hint_{i}")
    if c3.button("Remove", key=f"remove_{i}"):
        st.session_state.fields.pop(i)
        st.rerun()

if st.button("+ Add field"):
    st.session_state.fields.append({"name": "", "hint": ""})
    st.rerun()


# ---------------------------------------------------------------------------
# Step 2: upload PDFs
# ---------------------------------------------------------------------------
st.subheader("2. Upload contract PDFs")
uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def build_prompt(contract_text: str, fields: list) -> str:
    field_lines = "\n".join(f'- "{f["name"]}": {f["hint"]}' for f in fields if f["name"].strip())
    return f"""You are extracting key terms from a contract. Read the contract text below and
return ONLY a JSON object with exactly these keys:

{field_lines}

Rules:
- If a value isn't found in the text, use "Not found" — never guess or invent a value.
- Keep each value short (a phrase, date, or figure), not a full sentence.
- Return ONLY the JSON object, no other text, no markdown fences.

CONTRACT TEXT:
\"\"\"
{contract_text}
\"\"\"
"""


def call_claude(prompt: str, api_key: str, model: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def call_ollama(prompt: str, url: str, model: str) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_groq(prompt: str, api_key: str, model: str) -> str:
    return call_openai_compatible(prompt, "https://api.groq.com/openai/v1", api_key, model)


def call_openai_compatible(prompt: str, base: str, key: str, model: str) -> str:
    resp = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw: str) -> dict:
    # Strip markdown fences if the model added them despite instructions
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back: try to grab the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "Could not parse model response", "raw_response": raw[:500]}


# ---------------------------------------------------------------------------
# Step 3: run extraction
# ---------------------------------------------------------------------------
st.subheader("3. Run extraction")

active_fields = [f for f in st.session_state.fields if f["name"].strip()]
missing_key = (
    (backend == "Claude (Anthropic API)" and not claude_api_key)
    or (backend == "Groq" and not groq_api_key)
)

if missing_key:
    st.warning(f"Enter your {'Anthropic' if backend.startswith('Claude') else 'Groq'} API key in the sidebar to use this backend.")

if st.button("Extract key terms", type="primary", disabled=not (uploaded_files and active_fields) or missing_key):
    rows = []
    progress = st.progress(0.0, text="Starting...")

    for idx, file in enumerate(uploaded_files):
        progress.progress(idx / len(uploaded_files), text=f"Reading {file.name}...")
        try:
            text = extract_pdf_text(file)
        except Exception as e:
            rows.append({"File": file.name, "Error": f"Failed to read PDF: {e}"})
            continue

        if not text.strip():
            rows.append({"File": file.name, "Error": "No extractable text (likely a scanned/image PDF — needs OCR)"})
            continue

        truncated = text[:max_chars]
        prompt = build_prompt(truncated, active_fields)

        progress.progress((idx + 0.5) / len(uploaded_files), text=f"Extracting from {file.name}...")
        try:
            if backend == "Claude (Anthropic API)":
                raw = call_claude(prompt, claude_api_key, claude_model)
            elif backend == "Groq":
                raw = call_groq(prompt, groq_api_key, groq_model)
            elif backend == "Ollama (local)":
                raw = call_ollama(prompt, ollama_url, ollama_model)
            else:
                raw = call_openai_compatible(prompt, api_base, api_key, api_model)
        except Exception as e:
            rows.append({"File": file.name, "Error": f"LLM call failed: {e}"})
            continue

        parsed = parse_json_response(raw)
        row = {"File": file.name}
        row.update(parsed)
        rows.append(row)

    progress.progress(1.0, text="Done")
    st.session_state.results = pd.DataFrame(rows)

if st.session_state.results is not None:
    st.subheader("Results")
    st.dataframe(st.session_state.results, use_container_width=True)

    buffer = io.BytesIO()
    st.session_state.results.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="contract_key_terms.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
