"""
Contract Terms Extract — OpenAI-powered contract term extraction

Upload PDF contracts, define whatever fields you want extracted,
and get a table (+ Excel export) with the extracted values per document.

Run:
    pip install -r requirements.txt
    streamlit run app.py

API key:
    Create a file named `.env` in the same folder as this script with:
        OPENAI_API_KEY=sk-...
    Never commit this file to GitHub — keep it local only (see .gitignore).
    On Streamlit Cloud, set OPENAI_API_KEY under App settings > Secrets instead.
"""

import io
import json
import os
import re

import pandas as pd
import pdfplumber
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
MAX_CHARS = 80000  # contract text sent to the model per document (~20k tokens — comfortably covers most contracts)

st.set_page_config(page_title="Contract Terms Extract", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .app-title {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 2.1rem;
        color: #1F3A5F;
        margin-bottom: 0;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        color: #6B7280;
        font-size: 0.98rem;
        margin-top: 0.15rem;
        margin-bottom: 1.6rem;
    }
    .step-label {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: #1F3A5F;
        border-bottom: 1px solid #E4E7EB;
        padding-bottom: 0.4rem;
        margin-top: 1.6rem;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stStatusWidget"] { display: none; }
    #MainMenu, footer { visibility: hidden; }

    div.stButton > button[kind="primary"] {
        background-color: #1F3A5F;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #16283F;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    table th {
        background-color: #F4F6F8;
        color: #1F3A5F;
        text-align: left;
        padding: 0.5rem 0.7rem;
        border-bottom: 2px solid #E4E7EB;
    }
    table td {
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid #E4E7EB;
    }
    table tr:hover td {
        background-color: #FAFBFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# API key resolution — env var (.env) first, then Streamlit Secrets
# ---------------------------------------------------------------------------
def get_secret(key: str):
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError, st.errors.StreamlitAPIException):
        return None


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or get_secret("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">Contract Terms Extract</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload contracts, define the terms you need, get a clean summary table.</div>',
    unsafe_allow_html=True,
)

if not OPENAI_API_KEY:
    st.error(
        "No OpenAI API key configured. Add OPENAI_API_KEY to a local `.env` file, "
        "or under App settings \u2192 Secrets if this is deployed on Streamlit Cloud."
    )

with st.sidebar:
    st.caption("STATUS")
    if OPENAI_API_KEY:
        st.success("Connected")
    else:
        st.warning("Not connected")
    st.divider()
    st.caption(
        "Prototype — no data is stored between sessions. "
        "Contract text is sent to OpenAI's API for extraction."
    )


# ---------------------------------------------------------------------------
# Step 1: define fields
# ---------------------------------------------------------------------------
st.markdown('<div class="step-label">1 · Define the fields to extract</div>', unsafe_allow_html=True)

for i, field in enumerate(st.session_state.fields):
    c1, c2, c3 = st.columns([2, 5, 1])
    field["name"] = c1.text_input("Field name", field["name"], key=f"name_{i}", label_visibility="collapsed", placeholder="Field name")
    field["hint"] = c2.text_input("What to look for", field["hint"], key=f"hint_{i}", label_visibility="collapsed", placeholder="What to look for")
    if c3.button("Remove", key=f"remove_{i}"):
        st.session_state.fields.pop(i)
        st.rerun()

if st.button("+ Add field"):
    st.session_state.fields.append({"name": "", "hint": ""})
    st.rerun()


# ---------------------------------------------------------------------------
# Step 2: upload PDFs
# ---------------------------------------------------------------------------
st.markdown('<div class="step-label">2 · Upload contracts</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")


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


def _boxes_from_chars(chars):
    """Groups matched characters by visual line, returning one bbox per line
    so a match spanning a line-wrap highlights correctly instead of one box
    stretching across both lines."""
    lines = {}
    for c in chars:
        key = round(c["top"], 1)
        lines.setdefault(key, []).append(c)
    boxes = []
    for cs in lines.values():
        boxes.append((
            min(c["x0"] for c in cs),
            min(c["top"] for c in cs),
            max(c["x1"] for c in cs),
            max(c["bottom"] for c in cs),
        ))
    return boxes


def locate_quote(pdf_bytes: bytes, quote: str):
    """Search every page for the quote. Returns (page_number, [bbox, ...]) or None if not found.
    Multiple boxes are returned when the match spans a line wrap."""
    if not quote or not quote.strip():
        return None
    quote = quote.strip()
    words = quote.split()

    # Build a whitespace-tolerant regex so line wraps, extra spaces, or
    # non-breaking spaces in the PDF's text don't break an otherwise-correct match.
    def flexible_pattern(word_list):
        return r"\s+".join(re.escape(w) for w in word_list)

    # Try the full quote first, then progressively shorter windows (in case the
    # model's quote includes a word or two not present verbatim in the source).
    attempts = [words]
    if len(words) > 4:
        attempts.append(words[: len(words) - 2])
        attempts.append(words[2:])
    if len(words) > 6:
        attempts.append(words[2:-2])

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for word_set in attempts:
                    if not word_set:
                        continue
                    pattern = flexible_pattern(word_set)
                    try:
                        matches = page.search(pattern, regex=True, case=False)
                    except Exception:
                        matches = []
                    if matches:
                        boxes = _boxes_from_chars(matches[0]["chars"])
                        if boxes:
                            return page_num, boxes
    except Exception:
        return None
    return None


def render_highlighted_page(pdf_bytes: bytes, page_number: int, boxes) -> bytes:
    """Renders the given page as a PNG with each bbox in `boxes` highlighted."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[page_number]
        im = page.to_image(resolution=150)
        for bbox in boxes or []:
            im.draw_rect(bbox, fill=(255, 224, 102, 90), stroke=(230, 126, 14), stroke_width=2)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def build_prompt(contract_text: str, fields: list) -> str:
    field_lines = ",\n".join(
        f'    "{f["name"]}": {{"value": "<value — {f["hint"]}>", "source": "<short verbatim quote from the contract text that this value came from, 4-12 words, EXACT wording, or empty string if not found>"}}'
        for f in fields if f["name"].strip()
    )
    return f"""You are extracting key terms from a contract. Read the contract text below and
return ONLY a JSON object with exactly this shape:

{{
  "fields": {{
{field_lines}
  }},
  "pricing_type": "Fixed" | "Variable" | "Mixed" | "Not found",
  "pricing_notes": "any caps, minimum fees, or rate-review terms that affect the total (short phrase), or empty string if none",
  "channels": [
    {{
      "name": "the channel, service line, or unit this rate applies to (e.g. 'Paid Search (Search Ads)')",
      "name_source": "<short verbatim quote naming this channel, EXACT wording>",
      "billing_unit": "how it's billed (e.g. 'Per Click (CPC)')",
      "rate_value": <number only, e.g. 2.75>,
      "rate_currency": "currency code or symbol, e.g. AED",
      "rate_unit_label": "what the rate is per, e.g. 'click' or '1,000 impressions'",
      "rate_per_n_units": <if the rate is per N units like 'per 1,000 impressions' put N here, else 1>,
      "rate_source": "<short verbatim quote containing this rate, EXACT wording>",
      "volume_low": <lowest estimated volume as a plain number, e.g. 6000>,
      "volume_high": <highest estimated volume as a plain number; same as volume_low if only one figure is given>,
      "volume_source": "<short verbatim quote containing this volume figure, EXACT wording>"
    }}
  ]
}}

Rules:
- "fields" must contain exactly the keys listed above, each an object with "value" and "source". If a value isn't found, use "Not found" for value and "" for source — never guess or invent a value. Keep each value short (a phrase, date, or figure), not a full sentence.
- Every "source" quote must be copied EXACTLY from the contract text below — same words, same characters, same punctuation and capitalization — so it can be located verbatim on the page. Do not paraphrase or summarize it. Keep each quote short (under ~12 words) and specific enough to be unique in the document.
- Classify "pricing_type" as "Variable" if fees depend on usage/volume (e.g. per-click, per-impression, per-lead, commission-based), "Fixed" if it's a flat recurring or one-time fee, "Mixed" if both appear, "Not found" if unclear.
- Populate "channels" ONLY when the contract gives per-unit rates with an associated volume or volume estimate (e.g. a rate card, media plan, or pricing table). Leave it as an empty list if the contract is a flat/fixed fee with no such breakdown.
- Every number in "channels" must be a plain numeric value with no currency symbols, commas, or text — put units and currency in the separate label fields.
- Do not compute or total anything yourself — just extract the raw rate and volume figures per channel; the values will be calculated separately.
- Return ONLY the JSON object, no other text, no markdown fences.

CONTRACT TEXT:
\"\"\"
{contract_text}
\"\"\"
"""



def call_openai(prompt: str, api_key: str, model: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "Could not parse model response", "raw_response": raw[:500]}


def compute_channel_value(channel: dict):
    """Returns (estimated_value_number, display_string) or (None, 'n/a') if it can't be computed."""
    try:
        rate = float(channel.get("rate_value"))
        vol_low = float(channel.get("volume_low"))
        vol_high = float(channel.get("volume_high"))
        per_n = float(channel.get("rate_per_n_units") or 1) or 1
    except (TypeError, ValueError):
        return None, "n/a"

    mid_volume = (vol_low + vol_high) / 2
    value = rate * (mid_volume / per_n)
    currency = channel.get("rate_currency") or ""
    display = f"{currency} {value:,.0f}".strip()
    return value, display


def build_result_rows(file_name: str, parsed: dict, active_fields: list) -> list:
    if "error" in parsed:
        return [{"File": file_name, **parsed}]

    field_values = parsed.get("fields", {})
    base = {"File": file_name}
    for f in active_fields:
        entry = field_values.get(f["name"], {})
        value = entry.get("value", "Not found") if isinstance(entry, dict) else entry
        base[f["name"]] = value
    base["Pricing Type"] = parsed.get("pricing_type", "Not found")

    channels = parsed.get("channels") or []
    if not channels:
        base["Notes"] = parsed.get("pricing_notes", "")
        return [base]

    rows = []
    total_value = 0.0
    any_computed = False
    total_currency = ""
    for ch in channels:
        value, display = compute_channel_value(ch)
        if value is not None:
            total_value += value
            any_computed = True
            total_currency = total_currency or (ch.get("rate_currency") or "")
        rate_str = f"{ch.get('rate_currency', '')} {ch.get('rate_value', '')} / {ch.get('rate_unit_label', '')}".strip()
        vol_low, vol_high = ch.get("volume_low"), ch.get("volume_high")
        vol_str = f"{vol_low:,}".rstrip() if vol_low == vol_high else f"{vol_low:,} – {vol_high:,}" if vol_low is not None and vol_high is not None else "Not found"
        row = dict(base)
        row.update({
            "Channel": ch.get("name", "Not found"),
            "Billing Unit": ch.get("billing_unit", "Not found"),
            "Rate": rate_str,
            "Est. Volume": vol_str,
            "Est. Value": display,
        })
        rows.append(row)

    if any_computed:
        total_row = dict(base)
        notes = parsed.get("pricing_notes", "")
        total_row.update({
            "Channel": "TOTAL (all channels)",
            "Billing Unit": "",
            "Rate": "",
            "Est. Volume": "",
            "Est. Value": f"~ {total_currency} {total_value:,.0f}".strip(),
        })
        if notes:
            total_row["Est. Value"] += f"  (note: {notes})"
        rows.append(total_row)

    return rows


def build_source_items(file_name: str, parsed: dict, active_fields: list) -> list:
    """Flat list of {label, value, quote} for the click-to-verify panel."""
    if "error" in parsed:
        return []

    items = []
    field_values = parsed.get("fields", {})
    for f in active_fields:
        entry = field_values.get(f["name"], {})
        if isinstance(entry, dict):
            value, quote = entry.get("value", "Not found"), entry.get("source", "")
        else:
            value, quote = entry, ""
        items.append({"label": f["name"], "value": value, "quote": quote})

    for i, ch in enumerate(parsed.get("channels") or [], start=1):
        name = ch.get("name", f"Channel {i}")
        rate_display = f"{ch.get('rate_currency', '')} {ch.get('rate_value', '')} / {ch.get('rate_unit_label', '')}".strip()
        vol_display = f"{ch.get('volume_low', '')} – {ch.get('volume_high', '')}"
        items.append({"label": f"{name} — Name", "value": name, "quote": ch.get("name_source", "")})
        items.append({"label": f"{name} — Rate", "value": rate_display, "quote": ch.get("rate_source", "")})
        items.append({"label": f"{name} — Volume", "value": vol_display, "quote": ch.get("volume_source", "")})

    return items


# ---------------------------------------------------------------------------
# Step 3: run extraction
# ---------------------------------------------------------------------------
st.markdown('<div class="step-label">3 · Extract</div>', unsafe_allow_html=True)

active_fields = [f for f in st.session_state.fields if f["name"].strip()]

if "source_items" not in st.session_state:
    st.session_state.source_items = {}  # file_name -> list of {label, value, quote}
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = {}  # file_name -> raw bytes
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None  # (file_name, item_index)

if st.button(
    "Extract key terms",
    type="primary",
    disabled=not (uploaded_files and active_fields and OPENAI_API_KEY),
):
    rows = []
    source_items = {}
    pdf_bytes_map = {}
    truncation_warnings = []
    progress = st.progress(0.0, text="Starting...")

    for idx, file in enumerate(uploaded_files):
        progress.progress(idx / len(uploaded_files), text=f"Reading {file.name}...")
        pdf_bytes_map[file.name] = file.getvalue()
        try:
            text = extract_pdf_text(file)
        except Exception as e:
            rows.append({"File": file.name, "Error": f"Failed to read PDF: {e}"})
            continue

        if not text.strip():
            rows.append({"File": file.name, "Error": "No extractable text (likely a scanned/image PDF — needs OCR)"})
            continue

        if len(text) > MAX_CHARS:
            truncation_warnings.append(f"{file.name} ({len(text):,} characters — only the first {MAX_CHARS:,} were analyzed)")
        truncated = text[:MAX_CHARS]
        prompt = build_prompt(truncated, active_fields)

        progress.progress((idx + 0.5) / len(uploaded_files), text=f"Extracting from {file.name}...")
        try:
            raw = call_openai(prompt, OPENAI_API_KEY, OPENAI_MODEL)
        except Exception as e:
            rows.append({"File": file.name, "Error": f"Extraction failed: {e}"})
            continue

        parsed = parse_json_response(raw)
        rows.extend(build_result_rows(file.name, parsed, active_fields))
        source_items[file.name] = build_source_items(file.name, parsed, active_fields)

    progress.progress(1.0, text="Done")
    st.session_state.results = pd.DataFrame(rows)
    st.session_state.source_items = source_items
    st.session_state.pdf_bytes = pdf_bytes_map
    st.session_state.selected_source = None
    st.session_state.truncation_warnings = truncation_warnings

if st.session_state.results is not None:
    if st.session_state.get("truncation_warnings"):
        for w in st.session_state.truncation_warnings:
            st.warning(f"⚠️ {w} — terms appearing later in the document may have been missed.")

    st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)
    st.markdown(
        st.session_state.results.to_html(index=False, escape=False, na_rep=""),
        unsafe_allow_html=True,
    )

    buffer = io.BytesIO()
    st.session_state.results.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="contract_key_terms.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # -----------------------------------------------------------------
    # Verify a value against its source in the PDF
    # -----------------------------------------------------------------
    any_sources = any(st.session_state.source_items.get(fn) for fn in st.session_state.source_items)
    if any_sources:
        st.markdown('<div class="step-label">Verify a value</div>', unsafe_allow_html=True)
        st.caption("Click any extracted value below to see exactly where it came from in the contract.")

        for file_name, items in st.session_state.source_items.items():
            if not items:
                continue
            with st.expander(file_name, expanded=len(st.session_state.source_items) == 1):
                col_list, col_preview = st.columns([2, 3])
                with col_list:
                    for i, item in enumerate(items):
                        has_quote = bool(item["quote"])
                        st.button(
                            f"{item['label']}: {item['value']}" + ("" if has_quote else "  (no source found)"),
                            key=f"src_{file_name}_{i}",
                            use_container_width=True,
                            disabled=not has_quote,
                            on_click=(lambda fn=file_name, idx=i: st.session_state.__setitem__("selected_source", (fn, idx))) if has_quote else None,
                        )

                with col_preview:
                    sel = st.session_state.selected_source
                    if sel and sel[0] == file_name:
                        item = items[sel[1]]
                        pdf_bytes = st.session_state.pdf_bytes.get(file_name)
                        located = locate_quote(pdf_bytes, item["quote"]) if pdf_bytes else None
                        if located:
                            page_num, boxes = located
                            png = render_highlighted_page(pdf_bytes, page_num, boxes)
                            st.image(png, caption=f"Page {page_num + 1}", use_container_width=True)
                        else:
                            st.info(f"Couldn't pinpoint this on the page. Quoted text: \u201c{item['quote']}\u201d")
                    else:
                        st.caption("Select a value on the left to preview its source here.")
