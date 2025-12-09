import pandas as pd
import io

def clean_amount(val):
    """
    Cleans amount string (e.g. 1,00,000.00 Cr) to float.
    """
    if pd.isna(val):
        return 0.0
    val = str(val).lower().replace(',', '').replace('₹', '').strip()
    # Handle "Cr" or "Dr" if present in the amount cell itself
    if 'cr' in val:
        val = val.replace('cr', '').strip()
    if 'dr' in val:
        val = val.replace('dr', '').strip()
        # Note: Logic for negative/pos depends on column type, here we just want the number
    try:
        return float(val)
    except ValueError:
        return 0.0

def normalize_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the columns of the dataframe to a standard schema:
    [Date, Description, Amount, Category]
    
    Includes:
    1. Smart Header Detection (Scan first 10 rows)
    2. Strict Validation (Raise error if critical columns missing)
    3. Amount Cleaning (Convert to float)
    """
    
    # 1. Smart Header Detection
    # If the default header (row 0) doesn't contain "Date" or "Description", scan down
    potential_headers = ["date", "txn date", "transaction date", "description", "narration"]
    
    header_row_idx = 0
    # Check if current columns match
    current_cols = df_raw.columns.str.lower().str.strip().tolist()
    if not any(k in current_cols for k in potential_headers):
        # Scan first 10 rows
        for i in range(min(10, len(df_raw))):
            row_vals = df_raw.iloc[i].astype(str).str.lower().str.strip().tolist()
            if any(k in row_vals for k in potential_headers):
                header_row_idx = i + 1 # +1 because iloc[i] becomes header, data starts after
                # We need to reload or re-set header. 
                # Simpler to just assume the dataframe passed might be messy, 
                # but if we already have it as a DF, we can promote the row.
                new_header = df_raw.iloc[i]
                df_raw = df_raw.iloc[i+1:].reset_index(drop=True)
                df_raw.columns = new_header
                break
    
    df = df_raw.copy()
    
    # 2. Iterate and Rename
    standard_columns = {
        'Date': ['txn date', 'transaction date', 'date', 'value date'],
        'Description': ['description', 'narration', 'particulars', 'remarks', 'details'],
        'Amount': ['withdrawal', 'debit', 'amount', 'dr'], # Priority to Debit/Withdrawal
        'Credit': ['deposit', 'credit', 'cr'] # We might need this to calculate net
    }

    df.columns = df.columns.astype(str).str.lower().str.strip()
    
    renamed_map = {}
    for standard, keywords in standard_columns.items():
        for col in df.columns:
            if col in keywords:
                renamed_map[col] = standard
                break 
    
    if renamed_map:
        df = df.rename(columns=renamed_map)
    
    # 3. Strict Validation
    required_cols = ['Date', 'Description']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Normalization Failed: Could not find columns: {missing}. Please check the file.")
        
    # 4. Amount Logic (Handle Debit vs Credit if split)
    if 'Amount' not in df.columns:
        if 'Credit' in df.columns:
             # If we have Credit but not Amount/Debit, assume Amount = Credit (Income) or similar??
             # Actually, usually we want "Amount" to be the spending.
             # If we have Debit and Credit, we should unify them?
             # For PII/Module A, we just want a standard table.
             # Let's say Amount = Debit - Credit or just keep them?
             # For now, if Amount (Debit) is missing but we have Credit, let's allow it but warn.
             pass
        else:
             raise ValueError("Normalization Failed: Could not find 'Amount' or 'Debit' column.")

    # 5. Clean Data Types
    # Clean Amount/Debit/Credit
    cols_to_clean = [c for c in ['Amount', 'Credit'] if c in df.columns]
    for col in cols_to_clean:
         df[col] = df[col].apply(clean_amount)
         
    return df
