# WealthWise-AI---Outskill-Capstone-Project
WealthWise AI is an end-to-end personal finance system that acts as an AI-CFO. It analyzes financial documents, tracks expenses, decodes tax rules with RAG, processes statements via OCR/vision models, and uses multi-agent workflows to optimize spending, detect anomalies, and give proactive financial advice.


Module A Walkthrough: Secure Data Ingestion
This document details the features, usage, and verification of Module A (Secure Data Ingest) for Wealthwise AI.

1. Overview
Module A is the entry point for financial data. It securely ingests, cleans, and anonymizes Bank Statements and Salary Slips before any analysis occurs.

2. Features Implemented
Multi-File Upload:
  - Support for uploading multiple CSV bank statements simultaneously.
Automatic consolidation of data from multiple files.
Salary Slip Processing:
Dedicated tab for PDF upload.
Automatic text extraction using pypdf.
PII Scrubbing on extracted text.
Robustness:
Smart Header Detection: Scans the first 10 rows to locate the valid header row, ignoring metadata like bank addresses.
Schema Validation: Ensures Date and Description columns exist.
Amount Cleaning: Converts formatted strings (e.g., "1,00,000.00 Cr") to floats.
Privacy (PII Scrubbing):
Redacts Names, Phone Numbers, and Emails using Microsoft Presidio.
Date Protection: Explicit logic to preserve dates (e.g., "01-10-2025") while scrubbing other PII.
3. How to Use
Start the App:
streamlit run app.py
Bank Statements (Tab 1):
Click "Browse files".
Select one or more CSV files (e.g., 
Rohan_Bank_Statement.csv
).
Click "Process Bank Statements".
View the "Sanitized Data Preview" to confirm names are redacted ([REDACTED]).
Salary Slips (Tab 2):
Switch to the "Salary Slips" tab.
Upload a PDF file.
Click "Process & Scrub".
Expand the "Preview" to see the redacted text content.
4. Verification Results
Automated verification script 
verify_robustness.py
 confirmed:

Normalization: Successfully identified headers in messy CSVs.
Data Types: Amount column correctly converted to float.
Privacy: High-confidence PII removed, Dates preserved.
5. Files Created
app.py
: Main Streamlit interface.
utils/normalizer.py
: Logic for smart header detection and schema mapping.
utils/sanitizer.py
: PII scrubbing logic using Presidio.
requirements.txt
: Dependencies (presidio, pypdf, streamlit).
