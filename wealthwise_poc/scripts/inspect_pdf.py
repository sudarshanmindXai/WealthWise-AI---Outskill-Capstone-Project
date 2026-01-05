
from pypdf import PdfReader
import sys

def inspect_pdf(pdf_path, start_page=10, num_pages=5):
    try:
        reader = PdfReader(pdf_path)
        print(f"Total Pages: {len(reader.pages)}")
        
        for i in range(start_page, min(start_page + num_pages, len(reader.pages))):
            page = reader.pages[i]
            text = page.extract_text()
            print(f"--- PAGE {i} ---")
            print(text)
            print("----------------")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf_path = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income-tax-act-1961-as-amended-by-finance-act-2025.pdf"
    inspect_pdf(pdf_path, start_page=100, num_pages=5)
