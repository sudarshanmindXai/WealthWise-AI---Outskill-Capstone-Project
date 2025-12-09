import streamlit as st
import pandas as pd
from utils.normalizer import normalize_schema
from utils.sanitizer import clean_and_redact, redact_text
from pypdf import PdfReader
import io

st.set_page_config(page_title="Wealthwise AI - Secure Ingest", layout="wide")

st.title("Wealthwise AI - Secure Data Ingest (Module A)")
st.markdown("""
> **Privacy First**: Your data is processed locally in memory. 
> We analyze it, redact personal info, and then it's ready for the AI. 
> No raw data is saved to disk.
""")

# Initialize session state for data
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = None
if 'salary_data' not in st.session_state:
    st.session_state['salary_data'] = [] # List of {"filename": str, "text": str}

# Tabs for different upload types
tab1, tab2 = st.tabs(["🏦 Bank Statements", "📄 Salary Slips"])

with tab1:
    st.header("Step 1: Upload Bank Statements")
    # 1. File Uploader for Bank Statements (Multi-file)
    uploaded_files = st.file_uploader("Upload your Bank Statements (CSV)", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Bank Statements"):
            all_dfs = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    df = pd.read_csv(file)
                    
                    # Normalize
                    try:
                        df_normalized = normalize_schema(df.copy()) 
                        status.write("✅ Columns Standardized to: " + ", ".join(df_normalized.columns))
                    except ValueError as ve:
                        st.error(f"Schema Error: {ve}")
                        st.stop()

                    # Redact
                    df_cleaned = clean_and_redact(df_normalized)
                    
                    # Add source filename for tracking
                    df_cleaned['Source_File'] = file.name
                    all_dfs.append(df_cleaned)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Combine all files
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    st.success(f"Successfully processed {len(uploaded_files)} files!")
                    
                    st.subheader("Sanitized Data Preview")
                    st.dataframe(combined_df.head())
                    
                    st.session_state['financial_data'] = combined_df
                    st.info("Data stored in secure session state.")
                
            except Exception as e:
                st.error(f"Error processing files: {e}")

with tab2:
    st.header("Step 2: Upload Salary Slips")
    st.info("Supported: PDF (Text will be extracted and redacted)")
    # 2. File Uploader for Salary Slips (PDF Only for now as we added pypdf)
    salary_files = st.file_uploader("Upload Salary Slips (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if salary_files:
        if st.button("Process & Scrub Salary Slips"):
            processed_data = [] # Store scrubbed text
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                 for i, file in enumerate(salary_files):
                    status_text.text(f"Scanning {file.name} for PII...")
                    
                    # Extract Text
                    reader = PdfReader(file)
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text() + "\n"
                    
                    # Redact Text
                    scrubbed_text = redact_text(full_text)
                    
                    processed_data.append({
                        "filename": file.name,
                        "content": scrubbed_text
                    })
                    
                    progress_bar.progress((i + 1) / len(salary_files))
                 
                 st.session_state['salary_data'] = processed_data
                 st.success(f"Successfully scrubbed {len(salary_files)} salary slips.")
                 
                 # Show Preview
                 st.subheader("Scrubbed Text Preview")
                 if processed_data:
                     # Show first file content preview
                     with st.expander(f"Preview: {processed_data[0]['filename']}"):
                         st.text(processed_data[0]['content'][:1000] + "...") # First 1000 chars
            
            except Exception as e:
                st.error(f"Error processing salary slips: {e}")

# Sidebar for Debugging
with st.sidebar:
    st.header("Debug Info")
    
    st.subheader("Bank Data")
    if st.session_state['financial_data'] is not None:
        st.write(f"Rows: {len(st.session_state['financial_data'])}")
    else:
        st.write("Empty")
        
    st.subheader("Salary Data")
    if st.session_state['salary_data']:
        st.write(f"Files Processed: {len(st.session_state['salary_data'])}")
    else:
        st.write("No files")
