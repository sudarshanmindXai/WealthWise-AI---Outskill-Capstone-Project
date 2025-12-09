import pandas as pd
from utils.normalizer import normalize_schema
from utils.sanitizer import clean_and_redact, redact_text
# from pypdf import PdfWriter # We might not have PdfWriter installed or want to depend on it for test generation unless needed
# Let's just test the redact_text function which is the core logic used by the PDF processor

def verify():
    print("--- Starting Verification of Module A (Enhanced + Salary) ---")
    
    # 1. Test Redaction on Raw Text (Simulating PDF Content)
    print("\n[Testing Text Redaction]")
    sample_text = """
    Salary Slip for October 2025
    Name: Rohan Gupta
    Employee ID: 12345
    Phone: 9876543210
    Net Pay: 1,62,133
    Date: 01-10-2025
    """
    
    print("Original Text Snippet:")
    print(sample_text.strip())
    
    scrubbed = redact_text(sample_text)
    
    print("\nScrubbed Text Snippet:")
    print(scrubbed.strip())
    
    if "[REDACTED]" in scrubbed:
        print("✅ Pass: PII Redacted in text block.")
    else:
        print("❌ FAILED: PII NOT Redacted.")
        
    if "Rohan Gupta" in scrubbed:
        print("❌ FAILED: Name still present.")
    else:
        print("✅ Pass: Name removed.")
        
    if "01-10-2025" in scrubbed: # Date should be preserved if possible, but Presidio might catch it if we don't handle it.
        # Note: clean_and_redact handles the 'Date' COLUMN skipping. 
        # But redact_text on raw text might scrub dates if they look like phones or if Presidio is aggressive.
        # Let's see what happens.
        print("ℹ️ Note: Date preserved in text block (Ideal but might be hard in unstructured text without context).")
    else:
        print("ℹ️ Note: Date redacted (Acceptable for unstructured text if it looks like PII, but check if it was intended).")

if __name__ == "__main__":
    verify()
