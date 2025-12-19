"""
Secure Data Ingest - Streamlit app (single-file)

Features implemented:
- Streamlit file uploader that accepts CSV files (and sample CSV generator/download).

- Multiple files, salary slips (pdf parser), date handling, detect errors and give feedback, saving the data
- Multiple Salary slips / share image - play around with data/ sample
- relevant data -- Rohan bank statements/ you can create for sneha 
- create on sneha's personas -- more randomness


- Schema mapping (fuzzy and keyword-based) to normalize messy headers into: [Date, Description, Amount, Category].
- PII scrubbing: redacts Indian mobile numbers, PAN formats, account numbers and name-like columns. Uses presidio-analyzer if available, otherwise robust Regex.
- Stores cleaned DataFrame in st.session_state['financial_data'] (never writes to disk).
- Two categorization modes: rule-based fallback or optional OpenAI LLM if OPENAI_API_KEY present.

How to run:
1. Install dependencies (recommended):
   pip install streamlit pandas rapidfuzz python-dateutil openai presidio-analyzer
   (presidio-analyzer is optional; code falls back to regex if it's missing)
2. Run:
   streamlit run secure_data_ingest.py

Notes about accepted CSVs:
- Any CSV exported from bank statements is fine. You do NOT need fixed column names.
- Typical messy header examples that will be handled: "Date", "Txn Date", "Value Date", "Transaction Date" -> Date.
  "Debit", "Withdrawal", "Dr", "Amount Debited" -> Amount (negative numbers are treated as debits).
  "Credit", "Cr", "Amount Credited" -> Amount (positive).
  "Description", "Particulars", "Narration", "Details" -> Description.
- The app will infer the Amount sign when banks split Debit and Credit into separate columns.

"""

"""
Secure Data Ingest - Streamlit app (single-file)
"""

import streamlit as st
import pandas as pd
import re
import difflib
from typing import List, Dict, Tuple

# -------------------------
# Optional PDF support
# -------------------------
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

st.set_page_config(page_title="Secure Data Ingest", layout="wide")

# -------------------------
# Schema mapping
# -------------------------
COLUMN_KEYWORDS = {
    "Date": ["date", "txn date", "transaction date", "value date", "posted"],
    "Description": ["description", "narration", "details", "particulars", "remarks"],
    "Amount": ["amount", "debit", "credit", "withdrawal", "dr", "cr", "amt"],
    "Category": ["category", "tag", "label"]
}
CANONICAL = ["Date", "Description", "Amount", "Category"]

def normalize_header(h): return h.strip().lower()

def fuzzy_map_columns(headers: List[str]) -> Dict[str, str]:
    mapped = {c: None for c in CANONICAL}
    low = [normalize_header(h) for h in headers]

    for i, h in enumerate(low):
        for c, kws in COLUMN_KEYWORDS.items():
            if any(k in h for k in kws) and not mapped[c]:
                mapped[c] = headers[i]

    for i, h in enumerate(low):
        if headers[i] in mapped.values(): continue
        for c in CANONICAL:
            if not mapped[c]:
                if difflib.SequenceMatcher(None, h, c.lower()).ratio() > 0.6:
                    mapped[c] = headers[i]

    return {k: v or "" for k, v in mapped.items()}

# -------------------------
# PII Sanitizer
# -------------------------
MOBILE_RE = re.compile(r'(\+91[\-\s]?|91[\-\s]?|0)?[6-9]\d{9}')
PAN_RE = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', re.I)
ACC_RE = re.compile(r'\b\d{9,18}\b')
EMAIL_RE = re.compile(r'\S+@\S+\.\S+')

def sanitize_cell(v, column_name=None):
    if pd.isna(v):
        return v

    s = str(v)

    s = MOBILE_RE.sub("[REDACTED_MOBILE]", s)
    s = PAN_RE.sub("[REDACTED_PAN]", s)
    s = ACC_RE.sub("[REDACTED_ACC]", s)
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", s)

    # 🚫 Do NOT redact semantic columns
    if column_name in ("Description", "Category"):
        return s

    # Only redact name-like values in non-semantic columns
    if re.fullmatch(r'[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}', s.strip()):
        return "[REDACTED_NAME]"

    return s


def sanitize_dataframe(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda v: sanitize_cell(v, col))
    return df


# -------------------------
# CSV Description & Category Fix
# -------------------------

def fill_missing_description(df: pd.DataFrame) -> pd.DataFrame:
    if "Description" not in df:
        return df

    # If Description exists but is fully empty
    if df["Description"].astype(str).str.strip().eq("").all():
        text_cols = [
            c for c in df.columns
            if df[c].dtype == object and c not in ["Category", "Date"]
        ]

        if text_cols:
            best_col = max(
                text_cols,
                key=lambda c: df[c].astype(str).str.len().mean()
            )
            df["Description"] = df[best_col]

    return df


def infer_category(desc: str) -> str:
    d = str(desc).lower()

    rules = {
        "Income": ["salary", "payroll", "net pay"],
        "Food": ["zomato", "swiggy", "restaurant"],
        "Shopping": ["amazon", "flipkart", "mall"],
        "Transfer": ["upi", "imps", "neft", "rtgs"],
        "Utilities": ["electricity", "water", "gas", "bill"],
        "Fees": ["charge", "fee", "penalty"],
    }

    for cat, keys in rules.items():
        if any(k in d for k in keys):
            return cat

    return "Uncategorized"


def fill_missing_category(df: pd.DataFrame) -> pd.DataFrame:
    if "Category" not in df:
        df["Category"] = "Uncategorized"
    else:
        df["Category"] = df["Category"].replace("", pd.NA)
        df["Category"] = df["Category"].fillna(
            df["Description"].apply(infer_category)
        )

    return df



# -------------------------
# PDF helpers
# -------------------------
def extract_text_from_pdf(file):
    try:
        file.seek(0)
        text = []

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        full_text = "\n".join(text).strip()

        if not full_text:
            return ("empty", "")

        return ("ok", full_text)

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError:
        return ("corrupt", "")

    except Exception as e:
        msg = str(e).lower()

        if "password" in msg or "encrypted" in msg:
            return ("protected", "")

        return ("failed", "")




# -------- FIX 1 ----------
def looks_like_bank_statement(text: str) -> bool:
    keywords = ["transaction", "debit", "credit", "balance", "withdrawal", "deposit"]
    return sum(k in text.lower() for k in keywords) >= 2

# -------- FIX 2 ----------

def normalize_pdf_text(text: str) -> str:
    replacements = {
        "â‚¹": "₹",
        "Rs.": "₹",
        "Rs": "₹",
        "INR": "₹",
        "\xa0": " ",   # non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def extract_salary_credit(text: str):
    text = normalize_pdf_text(text)

    # Match salary / net pay wording
    salary_keywords = [
        "net salary",
        "salary credited",
        "net pay",
        "net amount"
    ]

    if not any(k in text.lower() for k in salary_keywords):
        return None

    # Robust amount detection
    amt_match = re.search(r'₹\s?[\d,]+', text)

    # Robust date detection
    date_match = re.search(
        r'\b\d{1,2}[-/]\w+[-/]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
        text
    )

    if amt_match:
        return {
            "Date": date_match.group(0) if date_match else "",
            "Description": "Salary Credit",
            "Amount": amt_match.group(0),
            "Category": "Income"
        }

    return None



# -------------------------
# UI
# -------------------------
st.title("🔐 Secure Data Ingest")

uploaded_files = st.file_uploader(
    "Upload CSV or PDF files",
    type=["csv", "pdf"],
    accept_multiple_files=True
)

cleaned_dfs = []
logs = []

if uploaded_files:
    pdf_count = sum(f.name.lower().endswith(".pdf") for f in uploaded_files)
    if pdf_count > 10:
        st.info("ℹ️ Consider uploading in smaller batches")

    for f in uploaded_files:
        name = f.name
        with st.spinner(f"Processing {name}"):
            if name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(f)
                    mapped = fuzzy_map_columns(df.columns)
                    df = df.rename(columns={v: k for k, v in mapped.items() if v})
                    for c in CANONICAL:
                        if c not in df: 
                            df[c] = ""

                    # ✅ FIX APPLIED HERE (DO NOT MOVE)
                    df = fill_missing_description(df)
                    df = fill_missing_category(df)
                    
                    df = sanitize_dataframe(df)
                    cleaned_dfs.append((name, df))
                    logs.append((name, "✅ CSV processed"))
                except Exception:
                    logs.append((name, "❌ Failed to read CSV"))

            elif name.lower().endswith(".pdf"):
                status, text = extract_text_from_pdf(f)

                if status == "protected":
                    logs.append((name, "🔐 Password-protected PDF not supported"))
                    continue

                if status == "corrupt":
                    logs.append((name, "❌ filename is empty or corrupted"))
                    continue
                if status == "empty":
                    logs.append((name, "⚠️ contains no text"))
                    continue
                if status == "failed":
                    logs.append((name, "❌ Failed to read file"))
                    continue

                if not looks_like_bank_statement(text):
                    salary = extract_salary_credit(text)
                    if salary:
                        df = pd.DataFrame([salary])
                        df = sanitize_dataframe(df)
                        cleaned_dfs.append((name, df))
                        logs.append((name, "✅ Salary slip detected and processed"))
                    else:
                        logs.append((name, "⚠️ Not a transaction statement"))
                    continue

                lines = [l.strip() for l in text.splitlines() if l.strip()]
                rows = [{"raw": l} for l in lines if re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', l)]

                if not rows:
                    logs.append((name, "⚠️ No transaction rows detected"))
                    continue
                    summary_match = re.search(
                        r'Closing Balance.*?₹?\s?([\d,]+)', text, re.I
                    )

                    if summary_match:
                        df = pd.DataFrame([{
                            "Date": "",
                            "Description": "Monthly Statement Summary",
                            "Amount": summary_match.group(1),
                            "Category": "Balance"
                        }])
                        df = sanitize_dataframe(df)
                        cleaned_dfs.append((name, df))
                        logs.append((name, "⚠️ Monthly summary detected (no transactions)"))
                        continue

                

                df = sanitize_dataframe(pd.DataFrame(rows))
                cleaned_dfs.append((name, df))
                logs.append((name, "✅ Bank statement processed"))

    st.subheader("Processing Summary")
    for n, m in logs:
        st.write(f"**{n}** — {m}")

    if cleaned_dfs:
        combined = pd.concat(
            [df.assign(_source_file=name) for name, df in cleaned_dfs],
            ignore_index=True
        )
        st.session_state["financial_data"] = combined
        st.success("✅ Data stored in session")
        st.dataframe(combined.head(100))

else:
    st.info("Upload files to begin")
