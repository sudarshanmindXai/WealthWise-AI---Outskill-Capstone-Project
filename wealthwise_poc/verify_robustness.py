import pandas as pd
from utils.normalizer import normalize_schema

def verify_robustness():
    print("--- Starting Robustness Verification ---")
    
    # 1. Load the "Complex" CSV (with metadata rows)
    csv_path = "Rohan_Bank_Statement.csv"
    print(f"Loading {csv_path} (Raw read)...")
    
    # Simulate how Streamlit reads it (just pd.read_csv without knowing skip_rows)
    # The first few rows will be garbage metadata.
    try:
        df_raw = pd.read_csv(csv_path, header=None) # Read no header initially to capture all rows
        # Actually standard pd.read_csv might pick row 0 as header.
        # Let's read it as is.
        df_raw = pd.read_csv(csv_path) 
    except Exception as e:
        print(f"Read Error: {e}")
        return

    print(f"Raw Shape: {df_raw.shape}")
    print("Raw Columns (likely garbage):", df_raw.columns.tolist()[:3])
    
    # 2. Test Smart Header Detection
    print("\n[Testing Smart Header Detection & Normalization]")
    try:
        df_norm = normalize_schema(df_raw)
        print("✅ Normalization Successful!")
        print("Normalized Columns:", df_norm.columns.tolist())
    except ValueError as e:
        print(f"❌ FAILED: {e}")
        return

    # 3. Test Amount Cleaning
    print("\n[Testing Amount Cleaning]")
    # Check if 'Amount' (or Credit/Debit) is float
    if 'Amount' in df_norm.columns:
        dtype = df_norm['Amount'].dtype
        print(f"Amount Type: {dtype}")
        if pd.api.types.is_float_dtype(dtype):
            print("✅ Pass: Amount is float.")
        else:
            print(f"❌ FAILED: Amount is {dtype} (expected float).")
            
        # Check a specific value
        sample_val = df_norm['Amount'].iloc[0]
        print(f"Sample Amount: {sample_val} (Type: {type(sample_val)})")
    
    # 4. Check Date
    print("\n[Testing Date]")
    if 'Date' in df_norm.columns:
        print(f"Sample Date: {df_norm['Date'].iloc[0]}")
    else:
        print("❌ FAILED: Date column missing.")

if __name__ == "__main__":
    verify_robustness()
