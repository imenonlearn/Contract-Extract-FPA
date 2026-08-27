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

import openpyxl
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


def apply_reviewer_overrides(results_df: pd.DataFrame) -> pd.DataFrame:
    """Applies any reviewer edits stored under override_{row}_{col} keys to a copy
    of the audit results — used both for the on-screen table and anywhere else
    (like Forecasting) that needs the reviewer's corrected values, not the raw model output."""
    display_df = results_df.copy()
    editable_cols = [c for c in display_df.columns if c not in ("File", "Error")]
    for row_idx in display_df.index:
        for col in editable_cols:
            override_key = f"override_{row_idx}_{col}"
            if override_key in st.session_state:
                display_df.at[row_idx, col] = st.session_state[override_key]
    return display_df


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
        # Clear any stale manual overrides from a previous extraction run —
        # row indices reset each run, so leftover overrides could apply to the wrong row.
        for k in list(st.session_state.keys()):
            if k.startswith("override_"):
                del st.session_state[k]

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
        st.session_state.reviewable_columns = [f["name"] for f in active_fields]

    if st.session_state.results is not None:
        if st.session_state.get("truncation_warnings"):
            for w in st.session_state.truncation_warnings:
                st.warning(f"⚠️ {w} — terms appearing later in the document may have been missed.")

        # Apply any reviewer edits — every cell (except File) can be corrected,
        # not just ones the model marked "Not found" — so both the table and
        # the Excel export always reflect the reviewer's current word on it.
        display_df = apply_reviewer_overrides(st.session_state.results)
        editable_cols = [c for c in display_df.columns if c not in ("File", "Error")]
        for row_idx in display_df.index:
            for col in editable_cols:
                override_key = f"override_{row_idx}_{col}"
                if override_key in st.session_state:
                    display_df.at[row_idx, col] = st.session_state[override_key]

        st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)
        st.markdown(
            display_df.to_html(index=False, escape=False, na_rep=""),
            unsafe_allow_html=True,
        )

        buffer = io.BytesIO()
        display_df.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            "Download as Excel",
            data=buffer.getvalue(),
            file_name="contract_key_terms.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # -----------------------------------------------------------------
        # Edit values — expand a row to correct anything, not just blanks
        # -----------------------------------------------------------------
        st.markdown('<div class="step-label">Edit values</div>', unsafe_allow_html=True)
        st.caption("Open a row to correct any value — whether it's missing or you simply disagree with what was extracted. Changes apply to the table and Excel export immediately.")
        for row_idx in display_df.index:
            file_label = display_df.at[row_idx, "File"] if "File" in display_df.columns else f"Row {row_idx}"
            channel_val = display_df.at[row_idx, "Channel"] if "Channel" in display_df.columns else None
            channel_suffix = f" — {channel_val}" if channel_val and str(channel_val).strip() else ""
            has_missing = any(
                str(st.session_state.results.at[row_idx, c]).strip().lower() == "not found"
                for c in editable_cols if c in st.session_state.results.columns
            )
            with st.expander(f"{file_label}{channel_suffix}" + ("  ⚠️ has missing values" if has_missing else "")):
                for col in editable_cols:
                    if col not in display_df.columns:
                        continue
                    override_key = f"override_{row_idx}_{col}"
                    st.text_input(
                        col,
                        value=str(st.session_state.results.at[row_idx, col]),
                        key=override_key,
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


def normalize_name(x) -> str:
    return str(x).strip().lower()


def build_monthly_actuals_lookup(df_dump, date_col, name_col, amount_col, year):
    """Aggregates a transaction-level actuals dump into {(normalized_name, month_num): amount}."""
    df = df_dump.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].dt.year == year]
    df["_name_norm"] = df[name_col].apply(normalize_name)
    df["_month"] = df[date_col].dt.month
    try:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    except Exception:
        pass
    grouped = df.groupby(["_name_norm", "_month"])[amount_col].sum()
    return grouped.to_dict()


CATEGORY_ABBREVIATIONS = list(CLASSIFICATION_CATEGORIES.keys())


def classify_row_color(cell) -> str:
    """Returns 'purple' (heading), 'grey' (subheading), or 'plain' (line item),
    based on the theme-based fill this budget template uses."""
    fg = cell.fill.fgColor
    if fg.type == "theme" and fg.theme == 8:
        return "purple"
    elif fg.type == "theme" and fg.theme == 0:
        return "grey"
    return "plain"


def build_sheet_structure(ws) -> list:
    """Walks column A from row 3 onward (skipping the title and month-header rows)
    and classifies each row as heading / subheading / line_item / total."""
    nodes = []
    for row in range(3, ws.max_row + 1):
        cell = ws.cell(row=row, column=1)
        name = cell.value
        if name is None or (isinstance(name, str) and not name.strip()):
            continue
        color = classify_row_color(cell)
        if isinstance(name, str) and name.strip().lower() == "total":
            node_type = "total"
        elif color == "purple":
            node_type = "heading"
        elif color == "grey":
            node_type = "subheading"
        else:
            node_type = "line_item"
        nodes.append({"type": node_type, "row": row, "name": name.strip() if isinstance(name, str) else name})
    return nodes


def get_hierarchy(nodes: list):
    """Builds the heading -> subheading -> line_item tree, plus a flat
    {normalized_line_item_name: (heading, subheading)} index for existence checks."""
    headings = []
    current_heading = None
    current_subheading = None
    line_item_index = {}
    total_row = None
    for node in nodes:
        if node["type"] == "heading":
            current_heading = {"name": node["name"], "row": node["row"], "subheadings": [], "direct_items": []}
            headings.append(current_heading)
            current_subheading = None
        elif node["type"] == "subheading":
            current_subheading = {"name": node["name"], "row": node["row"], "items": []}
            if current_heading is not None:
                current_heading["subheadings"].append(current_subheading)
        elif node["type"] == "line_item":
            key = normalize_name(node["name"])
            heading_name = current_heading["name"] if current_heading else None
            subheading_name = current_subheading["name"] if current_subheading else None
            line_item_index[key] = (heading_name, subheading_name)
            if current_subheading is not None:
                current_subheading["items"].append(node)
            elif current_heading is not None:
                current_heading["direct_items"].append(node)
        elif node["type"] == "total":
            total_row = node["row"]
    return headings, line_item_index, total_row


def parse_date_flexible(text):
    if not text or str(text).strip().lower() in ("not found", ""):
        return None
    try:
        result = pd.to_datetime(text, errors="coerce")
        return None if pd.isna(result) else result
    except Exception:
        return None


def parse_duration_months(term_text, effective_date=None):
    if not term_text or str(term_text).strip().lower() == "not found":
        return None
    text = str(term_text).lower()
    m = re.search(r"(\d+)\s*year", text)
    if m:
        return int(m.group(1)) * 12
    m = re.search(r"(\d+)\s*month", text)
    if m:
        return int(m.group(1))
    end_date = parse_date_flexible(term_text)
    if end_date is not None and effective_date is not None:
        months = (end_date.year - effective_date.year) * 12 + (end_date.month - effective_date.month)
        return max(months, 1)
    return None


def parse_amount(value_text):
    if not value_text or str(value_text).strip().lower() == "not found":
        return None
    text = str(value_text).replace(",", "")
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def compute_prorated_budget(contract_value, effective_date, duration_months, calendar_year):
    """Returns (total_for_year, monthly_rate, [active_month_numbers]) or (None, None, []) if unparseable."""
    if contract_value is None or effective_date is None or not duration_months:
        return None, None, []
    monthly_rate = contract_value / duration_months
    year_start = pd.Timestamp(year=calendar_year, month=1, day=1)
    year_end = pd.Timestamp(year=calendar_year, month=12, day=31)
    contract_end = effective_date + pd.DateOffset(months=duration_months)
    active_start = max(effective_date, year_start)
    active_end_excl = min(contract_end, year_end + pd.Timedelta(days=1))
    if active_end_excl <= active_start:
        return 0.0, monthly_rate, []
    months_active = []
    cursor = pd.Timestamp(year=active_start.year, month=active_start.month, day=1)
    while cursor < active_end_excl and cursor <= year_end:
        if cursor >= year_start:
            months_active.append(cursor.month)
        cursor += pd.DateOffset(months=1)
    total = monthly_rate * len(months_active)
    return total, monthly_rate, months_active


def build_placement_prompt(contract_text: str, headings: list) -> str:
    heading_lines = []
    for h in headings:
        if h["subheadings"]:
            subs = ", ".join(f'"{s["name"]}"' for s in h["subheadings"])
            heading_lines.append(f'  - "{h["name"]}" (subheadings: {subs})')
        else:
            heading_lines.append(f'  - "{h["name"]}" (no subheadings — items sit directly under it)')
    heading_block = "\n".join(heading_lines)
    return f"""A new contract needs to be placed into one of the existing budget categories below.
Read the contract text and pick the single best-fitting heading (and subheading, if that
heading has any) based on what the contract is actually for.

Existing categories:
{heading_block}

Return ONLY a JSON object of this shape:
{{
  "heading": "<one of the exact heading names above>",
  "subheading": "<one of that heading's exact subheading names, or empty string if the heading has no subheadings>"
}}

CONTRACT TEXT:
\"\"\"
{contract_text}
\"\"\"
"""


def find_insertion_row(heading: dict, subheading_name: str) -> int:
    """Row just after the last existing item in the target group (or right after
    the heading/subheading itself if that group has no items yet)."""
    if subheading_name:
        for sub in heading["subheadings"]:
            if sub["name"] == subheading_name:
                return (sub["items"][-1]["row"] + 1) if sub["items"] else (sub["row"] + 1)
        return heading["row"] + 1
    if heading["direct_items"]:
        return heading["direct_items"][-1]["row"] + 1
    if heading["subheadings"]:
        last_sub = heading["subheadings"][-1]
        return (last_sub["items"][-1]["row"] + 1) if last_sub["items"] else (last_sub["row"] + 1)
    return heading["row"] + 1


def write_forecast_workbook(wb, plan_by_sheet: dict) -> bytes:
    """Inserts new line items per sheet (bottom-to-top so row numbers stay valid),
    copies formatting from a neighboring line-item row, writes the prorated Budget
    figure and its monthly spread, then rewrites SUM formulas for every
    heading/subheading/grand-total row so the workbook stays a live, editable file."""
    from openpyxl.utils import get_column_letter
    import copy as _copy

    MONTH_COL_START = 4  # column D = Jan

    for sheet_name, insertions in plan_by_sheet.items():
        if not insertions:
            continue
        ws = wb[sheet_name]

        # Insert bottom-to-top so earlier insertion points aren't shifted by later ones.
        for ins in sorted(insertions, key=lambda x: x["insertion_row"], reverse=True):
            row = ins["insertion_row"]
            ws.insert_rows(row, 1)
            template_row = row - 1 if row > 1 else row + 1
            for col in range(1, 17):
                src = ws.cell(row=template_row, column=col)
                dst = ws.cell(row=row, column=col)
                dst.font = _copy.copy(src.font)
                dst.fill = _copy.copy(src.fill)
                dst.border = _copy.copy(src.border)
                dst.number_format = src.number_format
                dst.alignment = _copy.copy(src.alignment)

            ws.cell(row=row, column=1).value = ins["counterparty"]
            ws.cell(row=row, column=2).value = round(ins["prorated_total"], 2) if ins["prorated_total"] else 0
            for m in range(1, 13):
                col = MONTH_COL_START + (m - 1)
                ws.cell(row=row, column=col).value = round(ins["monthly_rate"], 2) if m in ins["active_months"] else 0

        # Recompute the hierarchy now that rows have shifted, and rewrite rollup formulas.
        nodes = build_sheet_structure(ws)
        headings, _, total_row = get_hierarchy(nodes)

        heading_cell_refs = []
        for h in headings:
            if h["subheadings"]:
                sub_refs = []
                for sub in h["subheadings"]:
                    if sub["items"]:
                        first, last = sub["items"][0]["row"], sub["items"][-1]["row"]
                        for col in [2] + list(range(4, 16)):
                            letter = get_column_letter(col)
                            ws.cell(row=sub["row"], column=col).value = f"=SUM({letter}{first}:{letter}{last})"
                    sub_refs.append(sub["row"])
                for col in [2] + list(range(4, 16)):
                    letter = get_column_letter(col)
                    formula = "=" + "+".join(f"{letter}{r}" for r in sub_refs) if sub_refs else 0
                    ws.cell(row=h["row"], column=col).value = formula
            elif h["direct_items"]:
                first, last = h["direct_items"][0]["row"], h["direct_items"][-1]["row"]
                for col in [2] + list(range(4, 16)):
                    letter = get_column_letter(col)
                    ws.cell(row=h["row"], column=col).value = f"=SUM({letter}{first}:{letter}{last})"
            heading_cell_refs.append(h["row"])

        if total_row is None:
            total_row = ws.max_row + 1
            ws.cell(row=total_row, column=1).value = "Total"
        for col in [2] + list(range(4, 16)):
            letter = get_column_letter(col)
            formula = "=" + "+".join(f"{letter}{r}" for r in heading_cell_refs) if heading_cell_refs else 0
            ws.cell(row=total_row, column=col).value = formula

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_actuals_upload_mode():
    st.caption("Upload a transaction-level actuals export here — Forecasting mode will automatically use it instead of the budget file's own month columns.")

    existing = st.session_state.get("actuals_dump_mapping")
    if existing:
        df_dump, date_col, dump_name_col, amount_col = existing
        st.success(f"Actuals dump loaded — {len(df_dump):,} rows, matched on '{dump_name_col}' with amounts from '{amount_col}'.")
        if st.button("Clear uploaded actuals dump"):
            st.session_state.pop("actuals_dump_mapping", None)
            st.rerun()
        st.markdown('<div class="step-label">Replace it</div>', unsafe_allow_html=True)

    dump_file = st.file_uploader("Actuals dump (.xlsx or .csv)", type=["xlsx", "csv"], key="actuals_dump_file")

    if not dump_file:
        if not existing:
            st.info("Upload a file with one row per transaction, with a date, a line-item name, and an amount.")
        return

    try:
        if dump_file.name.lower().endswith(".csv"):
            df_dump = pd.read_csv(dump_file)
        else:
            df_dump = pd.read_excel(dump_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")
        return

    if df_dump.empty:
        st.warning("This file appears to be empty.")
        return

    st.markdown('<div class="step-label">Confirm columns</div>', unsafe_allow_html=True)
    dump_columns = list(df_dump.columns)
    guessed_date = guess_column(dump_columns, ["date", "posted", "transaction"])
    guessed_dump_name = guess_column(dump_columns, ["name", "item", "channel", "department", "category", "line", "vendor"])
    guessed_amount = guess_column(dump_columns, ["amount", "value", "cost", "spend", "debit"])

    dc1, dc2, dc3 = st.columns(3)
    date_col = dc1.selectbox("Date column", dump_columns, index=dump_columns.index(guessed_date) if guessed_date in dump_columns else 0)
    dump_name_col = dc2.selectbox("Line item column", dump_columns, index=dump_columns.index(guessed_dump_name) if guessed_dump_name in dump_columns else 0)
    amount_col = dc3.selectbox("Amount column", dump_columns, index=dump_columns.index(guessed_amount) if guessed_amount in dump_columns else 0)

    st.session_state["actuals_dump_mapping"] = (df_dump, date_col, dump_name_col, amount_col)
    st.caption("Matched against your budget file's line items by name in Forecasting mode — go there once you're happy with the mapping above.")


def render_forecast_mode():
    st.caption("Adds new contracts into your Hub71 budget template, pro-rated for the months remaining in the calendar year, and keeps every heading/subheading/grand-total roll-up in sync.")

    if st.session_state.results is None:
        st.info("Run **Contract Audit** first — Forecasting needs the Counterparty, Effective Date, Term, and Contract Value from there.")
        return
    if not st.session_state.get("classification_results"):
        st.info("Run **Contract Classification** first — Forecasting uses the category (HC/CO/CS/GS/MM) to pick the right sheet for each contract.")
        return

    st.markdown('<div class="step-label">1 · Upload your budget workbook</div>', unsafe_allow_html=True)
    budget_file = st.file_uploader("Excel file (.xlsx)", type=["xlsx"], label_visibility="collapsed", key="forecast_budget_file")
    if not budget_file:
        st.info("Upload the Hub71 budget workbook (sheets named CO / GS / CS / MM, with purple heading rows and grey subheading rows).")
        return

    try:
        wb = openpyxl.load_workbook(budget_file, data_only=False)
    except Exception as e:
        st.error(f"Couldn't read this workbook: {e}")
        return

    year = st.number_input("Calendar year this budget covers", min_value=2020, max_value=2100, value=2026, step=1)

    # Build the sheet structure once so we can match categories to real sheets and show existing headings.
    sheet_data = {}
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        nodes = build_sheet_structure(ws)
        headings, line_item_index, total_row = get_hierarchy(nodes)
        sheet_data[sheet_name] = {"headings": headings, "line_item_index": line_item_index, "total_row": total_row}

    audit_df = apply_reviewer_overrides(st.session_state.results)
    pdf_bytes_map = st.session_state.get("pdf_bytes", {})

    st.markdown('<div class="step-label">2 · Review planned changes</div>', unsafe_allow_html=True)

    plan = []  # list of dicts, one per contract needing a decision
    for r in st.session_state.classification_results:
        file_name = r.get("file")
        if "error" in r:
            continue
        suggested_category = r.get("category", "")
        category = st.session_state.get(f"reviewer_class_{file_name}", suggested_category)
        if category not in sheet_data or sheet_data.get(category) is None:
            continue  # e.g. HC has no matching sheet in this workbook
        if category not in wb.sheetnames:
            plan.append({"file": file_name, "category": category, "status": "no_sheet"})
            continue

        file_rows = audit_df[audit_df.get("File") == file_name] if "File" in audit_df.columns else pd.DataFrame()
        if file_rows.empty:
            plan.append({"file": file_name, "category": category, "status": "no_audit_data"})
            continue
        row = file_rows.iloc[0]
        counterparty = str(row.get("Counterparty", "")).strip()
        if not counterparty or counterparty.lower() == "not found":
            plan.append({"file": file_name, "category": category, "status": "no_counterparty"})
            continue

        sheet_info = sheet_data[category]
        existing_match = sheet_info["line_item_index"].get(normalize_name(counterparty))
        if existing_match:
            plan.append({
                "file": file_name, "category": category, "counterparty": counterparty,
                "status": "already_present", "existing_heading": existing_match[0], "existing_subheading": existing_match[1],
            })
            continue

        eff_date = parse_date_flexible(row.get("Effective Date"))
        duration = parse_duration_months(row.get("Term / Expiry"), eff_date)
        value = parse_amount(row.get("Contract Value"))
        total, monthly_rate, active_months = compute_prorated_budget(value, eff_date, duration, year)

        # Ask the model to suggest where this fits among the sheet's real headings.
        suggested_heading, suggested_subheading = None, ""
        headings = sheet_info["headings"]
        if headings and OPENAI_API_KEY and file_name in pdf_bytes_map:
            try:
                text = extract_pdf_text(io.BytesIO(pdf_bytes_map[file_name]))[:MAX_CHARS]
                prompt = build_placement_prompt(text, headings)
                raw = call_openai(prompt, OPENAI_API_KEY, OPENAI_MODEL)
                parsed = parse_json_response(raw)
                if "heading" in parsed:
                    suggested_heading = parsed.get("heading")
                    suggested_subheading = parsed.get("subheading", "") or ""
            except Exception:
                pass
        if suggested_heading not in [h["name"] for h in headings]:
            suggested_heading = headings[0]["name"] if headings else None

        plan.append({
            "file": file_name, "category": category, "counterparty": counterparty,
            "status": "new", "sheet": category,
            "effective_date": eff_date, "duration_months": duration, "contract_value": value,
            "prorated_total": total, "monthly_rate": monthly_rate, "active_months": active_months,
            "suggested_heading": suggested_heading, "suggested_subheading": suggested_subheading,
        })

    if not plan:
        st.info("Nothing to plan yet — classify at least one contract into HC/CO/CS/GS/MM first.")
        return

    plan_by_sheet = {}
    for item in plan:
        label = f"{item.get('counterparty', item['file'])} — {item['category']}"
        with st.expander(label):
            if item["status"] == "no_sheet":
                st.warning(f"No sheet named '{item['category']}' in this workbook — can't place this contract.")
                continue
            if item["status"] == "no_audit_data":
                st.warning("No matching Contract Audit results found for this file.")
                continue
            if item["status"] == "no_counterparty":
                st.warning("Counterparty wasn't found in Contract Audit — fix that in Audit mode's 'Edit values' first.")
                continue
            if item["status"] == "already_present":
                loc = item["existing_heading"] + (f" → {item['existing_subheading']}" if item["existing_subheading"] else "")
                st.success(f"Already in the budget under **{loc}** — no new line will be added.")
                continue

            # status == "new"
            issues = []
            if item["effective_date"] is None:
                issues.append("Effective Date")
            if item["duration_months"] is None:
                issues.append("Term / Expiry")
            if item["contract_value"] is None:
                issues.append("Contract Value")
            if issues:
                st.warning(f"Couldn't parse: {', '.join(issues)} — enter the prorated amount manually below.")

            headings = sheet_data[item["category"]]["headings"]
            heading_names = [h["name"] for h in headings]
            default_h_idx = heading_names.index(item["suggested_heading"]) if item["suggested_heading"] in heading_names else 0
            chosen_heading_name = st.selectbox("Heading", heading_names, index=default_h_idx, key=f"plan_heading_{item['file']}")
            chosen_heading = next(h for h in headings if h["name"] == chosen_heading_name)

            chosen_subheading_name = ""
            if chosen_heading["subheadings"]:
                sub_names = [s["name"] for s in chosen_heading["subheadings"]]
                default_s_idx = sub_names.index(item["suggested_subheading"]) if item["suggested_subheading"] in sub_names else 0
                chosen_subheading_name = st.selectbox("Subheading", sub_names, index=default_s_idx, key=f"plan_sub_{item['file']}")

            default_total = round(item["prorated_total"], 2) if item["prorated_total"] is not None else 0.0
            reviewer_total = st.number_input(
                f"Budgeted amount for {year} (pro-rated)", value=default_total, key=f"plan_total_{item['file']}"
            )
            months_left = len(item["active_months"]) if item["active_months"] else 0
            monthly_rate = (reviewer_total / months_left) if months_left else 0.0
            st.caption(f"Spread across {months_left} active month(s) in {year}: ~{monthly_rate:,.2f}/month")

            plan_by_sheet.setdefault(item["category"], []).append({
                "counterparty": item["counterparty"],
                "insertion_row": find_insertion_row(chosen_heading, chosen_subheading_name),
                "prorated_total": reviewer_total,
                "monthly_rate": monthly_rate,
                "active_months": item["active_months"] or list(range(1, 13)),
            })

    st.markdown('<div class="step-label">3 · Apply</div>', unsafe_allow_html=True)
    if st.button("Apply to workbook", type="primary", disabled=not plan_by_sheet):
        workbook_bytes = write_forecast_workbook(wb, plan_by_sheet)
        st.session_state.forecast_workbook_bytes = workbook_bytes
        st.session_state.forecast_sheets_touched = list(plan_by_sheet.keys())

    if st.session_state.get("forecast_workbook_bytes"):
        st.markdown('<div class="step-label">Updated sheets</div>', unsafe_allow_html=True)
        result_wb = openpyxl.load_workbook(io.BytesIO(st.session_state.forecast_workbook_bytes), data_only=False)
        for sheet_name in st.session_state.forecast_sheets_touched:
            ws = result_wb[sheet_name]
            st.caption(f"**{sheet_name}**")
            rows_out = []
            for row in range(3, ws.max_row + 1):
                name = ws.cell(row=row, column=1).value
                if name is None:
                    continue
                budget = ws.cell(row=row, column=2).value
                month_vals = [ws.cell(row=row, column=4 + m).value for m in range(12)]
                rows_out.append([name, budget] + month_vals)
            preview_df = pd.DataFrame(rows_out, columns=["Line Item", "Budget"] + MONTH_NAMES)
            st.markdown(preview_df.to_html(index=False, escape=False, na_rep=""), unsafe_allow_html=True)

        st.download_button(
            "Download updated workbook",
            data=st.session_state.forecast_workbook_bytes,
            file_name=f"budget_updated_{year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Heading, subheading, and grand-total cells now contain live Excel SUM formulas, so the workbook stays fully editable.")

mode = st.radio(
    "Mode",
    ["Contract Audit", "Contract Classification", "Actuals Upload", "Forecasting"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

if mode == "Contract Audit":
    render_audit_mode()
elif mode == "Contract Classification":
    render_classification_mode()
elif mode == "Actuals Upload":
    render_actuals_upload_mode()
else:
    render_forecast_mode()
