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
def extract_pdf_text(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


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


def render_audit_mode():
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
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


CLASSIFICATION_CATEGORIES = {
    "HC": "Human Capital",
    "CO": "Incentives for Entrepreneurship",
    "CS": "Corporate Services",
    "GS": "Growth Strategy",
    "MM": "Marketing and Communication",
}


def build_classification_prompt(contract_text: str) -> str:
    category_lines = "\n".join(f'  - "{abbr}" = {full}' for abbr, full in CLASSIFICATION_CATEGORIES.items())
    return f"""You are classifying a contract along two dimensions. Read the contract text below and
return ONLY a JSON object with exactly this shape:

{{
  "pricing_classification": "Fixed" | "Variable" | "Mixed" | "Not found",
  "category": "HC" | "CO" | "CS" | "GS" | "MM",
  "category_reasoning": "<one short phrase, under 12 words, on why this category fits>"
}}

Rules:
- Classify "pricing_classification" as "Variable" if fees depend on usage/volume (e.g. per-click, per-impression, per-lead, commission-based), "Fixed" if it's a flat recurring or one-time fee, "Mixed" if both appear, "Not found" if unclear.
- Classify "category" as exactly one of the following abbreviations, based on the contract's subject matter and the nature of the services/goods being provided:
{category_lines}
- Pick the single best-fitting category even if the contract could plausibly touch more than one — choose based on the PRIMARY subject matter of the agreement.
- Return ONLY the JSON object, no other text, no markdown fences.

CONTRACT TEXT:
\"\"\"
{contract_text}
\"\"\"
"""


def render_classification_mode():
    pdf_bytes_map = st.session_state.get("pdf_bytes", {})

    if not pdf_bytes_map:
        st.info(
            "No contracts loaded yet. Upload and run extraction in **Contract Audit** mode first — "
            "Classification mode automatically reuses those same files."
        )
        return

    st.markdown('<div class="step-label">Contracts loaded from Contract Audit</div>', unsafe_allow_html=True)
    st.caption(", ".join(pdf_bytes_map.keys()))

    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None

    st.markdown('<div class="step-label">Classify</div>', unsafe_allow_html=True)
    if st.button(
        "Classify contracts",
        type="primary",
        disabled=not OPENAI_API_KEY,
    ):
        results = []
        file_names = list(pdf_bytes_map.keys())
        progress = st.progress(0.0, text="Starting...")

        for idx, file_name in enumerate(file_names):
            progress.progress(idx / len(file_names), text=f"Reading {file_name}...")
            try:
                text = extract_pdf_text(io.BytesIO(pdf_bytes_map[file_name]))
            except Exception as e:
                results.append({"file": file_name, "error": f"Failed to read PDF: {e}"})
                continue

            if not text.strip():
                results.append({"file": file_name, "error": "No extractable text (likely a scanned/image PDF)"})
                continue

            prompt = build_classification_prompt(text[:MAX_CHARS])
            progress.progress((idx + 0.5) / len(file_names), text=f"Classifying {file_name}...")
            try:
                raw = call_openai(prompt, OPENAI_API_KEY, OPENAI_MODEL)
            except Exception as e:
                results.append({"file": file_name, "error": f"Classification failed: {e}"})
                continue

            parsed = parse_json_response(raw)
            if "error" in parsed:
                results.append({"file": file_name, "error": parsed["error"]})
                continue

            results.append({
                "file": file_name,
                "pricing_classification": parsed.get("pricing_classification", "Not found"),
                "category": parsed.get("category", "") if parsed.get("category") in CLASSIFICATION_CATEGORIES else "",
                "category_reasoning": parsed.get("category_reasoning", ""),
            })

        progress.progress(1.0, text="Done")
        st.session_state.classification_results = results

    results = st.session_state.classification_results
    if not results:
        return

    st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)

    header = st.columns([2.2, 1.5, 1.8, 1.8])
    header[0].markdown("**File**")
    header[1].markdown("**Pricing Classification**")
    header[2].markdown("**Suggested Category**")
    header[3].markdown("**Reviewer Classification**")

    category_options = list(CLASSIFICATION_CATEGORIES.keys())
    export_rows = []

    for r in results:
        row = st.columns([2.2, 1.5, 1.8, 1.8])
        row[0].write(r["file"])

        if "error" in r:
            row[1].write("—")
            row[2].write("—")
            row[3].write(r["error"])
            export_rows.append({
                "File": r["file"], "Pricing Classification": "Error",
                "Suggested Category": "", "Reviewer Classification": "", "Notes": r["error"],
            })
            continue

        row[1].write(r["pricing_classification"])
        suggested = r["category"]
        suggested_label = f"{suggested} — {CLASSIFICATION_CATEGORIES[suggested]}" if suggested else "Not classified"
        row[2].write(suggested_label)

        default_index = category_options.index(suggested) if suggested in category_options else 0
        reviewer_choice = row[3].selectbox(
            "Reviewer override",
            category_options,
            index=default_index,
            key=f"reviewer_class_{r['file']}",
            label_visibility="collapsed",
            format_func=lambda abbr: f"{abbr} — {CLASSIFICATION_CATEGORIES[abbr]}",
        )

        export_rows.append({
            "File": r["file"],
            "Pricing Classification": r["pricing_classification"],
            "Suggested Category": suggested,
            "Reviewer Classification": reviewer_choice,
            "Notes": r.get("category_reasoning", ""),
        })

    st.caption("Reviewer Classification defaults to the suggested category — change it if the model got it wrong.")

    export_df = pd.DataFrame(export_rows)
    buffer = io.BytesIO()
    export_df.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="contract_classification.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
def guess_column(columns, keywords):
    for col in columns:
        norm = str(col).strip().lower()
        if any(kw in norm for kw in keywords):
            return col
    return None


def guess_month_columns(columns):
    """Returns a list of 12 column names (or None) matching Jan..Dec by prefix."""
    result = []
    for month in MONTH_NAMES:
        match = None
        for col in columns:
            norm = str(col).strip().lower()
            if norm.startswith(month.lower()):
                match = col
                break
        result.append(match)
    return result


def compute_forecast_rows(df, name_col, budget_col, month_cols, through_idx):
    """through_idx: number of months already actual (e.g. 8 = Jan-Aug are actuals).
    Returns a list of dicts with computed figures per row."""
    rows = []
    for _, row in df.iterrows():
        name = row[name_col]
        try:
            annual_budget = float(row[budget_col]) if pd.notna(row[budget_col]) else 0.0
        except (TypeError, ValueError):
            annual_budget = 0.0

        actual_to_date = 0.0
        for i in range(through_idx):
            col = month_cols[i]
            if col is None:
                continue
            try:
                val = float(row[col]) if pd.notna(row[col]) else 0.0
            except (TypeError, ValueError):
                val = 0.0
            actual_to_date += val

        remaining = annual_budget - actual_to_date
        months_left = 12 - through_idx
        monthly_forecast = (remaining / months_left) if months_left > 0 else 0.0

        rows.append({
            "name": name,
            "annual_budget": annual_budget,
            "actual_to_date": actual_to_date,
            "remaining": remaining,
            "months_left": months_left,
            "monthly_forecast": monthly_forecast,
        })
    return rows


def build_forecast_workbook(df, name_col, budget_col, month_cols, through_idx, forecast_rows, year) -> bytes:
    """Writes an updated workbook: actual months unchanged, remaining months filled
    with the evenly-spread forecast, with forecast cells highlighted."""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font

    wb = Workbook()
    ws = wb.active
    ws.title = f"Forecast {year}"

    forecast_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F3A5F", end_color="1F3A5F", fill_type="solid")

    headers = ["Line Item"] + MONTH_NAMES + ["Total", "Annual Budget", "Variance"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill

    monthly_totals = [0.0] * 12
    budget_total = 0.0

    for i, fr in enumerate(forecast_rows):
        row_values = [fr["name"]]
        month_values = []
        for m in range(12):
            if m < through_idx:
                col = month_cols[m]
                try:
                    val = float(df.iloc[i][col]) if col is not None and pd.notna(df.iloc[i][col]) else 0.0
                except (TypeError, ValueError):
                    val = 0.0
            else:
                val = round(fr["monthly_forecast"], 2)
            month_values.append(val)
            monthly_totals[m] += val
        row_total = sum(month_values)
        row_values += month_values
        row_values += [row_total, fr["annual_budget"], row_total - fr["annual_budget"]]
        ws.append(row_values)
        budget_total += fr["annual_budget"]

        excel_row = ws.max_row
        for m in range(through_idx, 12):
            ws.cell(row=excel_row, column=2 + m).fill = forecast_fill

    total_row = ["TOTAL"] + [round(v, 2) for v in monthly_totals] + [round(sum(monthly_totals), 2), round(budget_total, 2), round(sum(monthly_totals) - budget_total, 2)]
    ws.append(total_row)
    for cell in ws[ws.max_row]:
        cell.font = Font(bold=True)

    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = max(10, length + 2)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_forecast_mode():
    st.markdown('<div class="step-label">1 · Upload your budget workbook</div>', unsafe_allow_html=True)
    st.caption("This mode does pure arithmetic — no AI, no API calls, no cost.")
    budget_file = st.file_uploader("Excel file (.xlsx)", type=["xlsx"], label_visibility="collapsed")

    if not budget_file:
        st.info("Upload a workbook with one row per budget line, an annual budget column, and monthly columns (Jan\u2013Dec).")
        return

    try:
        xls = pd.ExcelFile(budget_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Couldn't read this workbook: {e}")
        return

    sheet_name = st.selectbox("Sheet", xls.sheet_names) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)

    if df.empty:
        st.warning("This sheet appears to be empty.")
        return

    st.markdown('<div class="step-label">2 · Confirm columns</div>', unsafe_allow_html=True)
    columns = list(df.columns)
    guessed_name = guess_column(columns, ["name", "item", "channel", "department", "category", "line"])
    guessed_budget = guess_column(columns, ["budget", "annual"])
    guessed_months = guess_month_columns(columns)

    c1, c2 = st.columns(2)
    name_col = c1.selectbox("Line item name column", columns, index=columns.index(guessed_name) if guessed_name in columns else 0)
    budget_col = c2.selectbox("Annual budget column", columns, index=columns.index(guessed_budget) if guessed_budget in columns else 0)

    st.caption("Month columns detected (Jan \u2192 Dec) \u2014 adjust any that are wrong:")
    month_cols = []
    cols_ui = st.columns(6)
    for i, month in enumerate(MONTH_NAMES):
        options = ["(none)"] + columns
        default = guessed_months[i] if guessed_months[i] in columns else "(none)"
        with cols_ui[i % 6]:
            picked = st.selectbox(month, options, index=options.index(default), key=f"month_col_{i}", label_visibility="visible")
        month_cols.append(None if picked == "(none)" else picked)

    st.markdown('<div class="step-label">3 · Forecast settings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    year = c1.number_input("Forecast year (calendar year, starts January)", min_value=2020, max_value=2100, value=2026, step=1)
    through_month = c2.selectbox("Actuals known through", MONTH_NAMES, index=7)
    through_idx = MONTH_NAMES.index(through_month) + 1

    if st.button("Generate forecast", type="primary"):
        if not budget_col or budget_col not in df.columns:
            st.error("Please select a valid annual budget column.")
            return
        if all(c is None for c in month_cols):
            st.error("At least one month column needs to be mapped.")
            return

        forecast_rows = compute_forecast_rows(df, name_col, budget_col, month_cols, through_idx)
        workbook_bytes = build_forecast_workbook(df, name_col, budget_col, month_cols, through_idx, forecast_rows, year)

        st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)
        preview = pd.DataFrame([{
            "Line Item": fr["name"],
            "Annual Budget": fr["annual_budget"],
            f"Actual (Jan\u2013{through_month})": fr["actual_to_date"],
            "Remaining Budget": fr["remaining"],
            f"Forecast / month ({12 - through_idx} left)": round(fr["monthly_forecast"], 2),
        } for fr in forecast_rows])
        st.markdown(preview.to_html(index=False, escape=False, na_rep=""), unsafe_allow_html=True)

        st.download_button(
            "Download updated workbook",
            data=workbook_bytes,
            file_name=f"budget_forecast_{year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---------------------------------------------------------------------------
# Mode toggle
# ---------------------------------------------------------------------------
mode = st.radio(
    "Mode",
    ["Contract Audit", "Contract Classification", "Forecasting"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

if mode == "Contract Audit":
    render_audit_mode()
elif mode == "Contract Classification":
    render_classification_mode()
else:
    render_forecast_mode()
