import re
import sys
import os
from pypdf import PdfReader

# Configuration
PDF_PATH = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income-tax-act-1961-as-amended-by-finance-act-2025.pdf"
OUTPUT_PATH = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_act.md"

# Regex Patterns
RE_HEADER = re.compile(r"INCOME-TAX ACT, 1961 - 2024", re.IGNORECASE)
RE_PAGE_NUM = re.compile(r"^\s*\d+\s+of\s+\d+.*", re.IGNORECASE)
RE_PAGE_NUM_SIMPLE = re.compile(r"^\s*\d+\s*$", re.IGNORECASE)

# Hierarchical Patterns
# Chapter: "CHAPTER I", "CHAPTER IV-D"
RE_CHAPTER = re.compile(r"^\s*CHAPTER\s+([IXVLCDM]+[-A-Z]*)\s*$", re.IGNORECASE)

# Section: "10.", "10A.", "115BAC."
# Must take care not to match "(1)" as a section.
# Sections usually start a line, are numbers followed by optional chars and a dot.
RE_SECTION = re.compile(r"^\s*(\d+[A-Z]*)\.\s+(.*)")

# Omitted/Repealed
RE_OMITTED = re.compile(r"^.*\[.*Omitted.*by.*\].*", re.IGNORECASE)

def clean_line(line):
    """Removes headers, footers, and extra whitespace."""
    if RE_HEADER.search(line):
        return None
    if RE_PAGE_NUM.search(line):
        return None
    if RE_PAGE_NUM_SIMPLE.match(line):
        return None
    return line.strip()

def main():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at {PDF_PATH}")
        sys.exit(1)

    print(f"Reading PDF from {PDF_PATH}...")
    try:
        reader = PdfReader(PDF_PATH)
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        sys.exit(1)

    total_pages = len(reader.pages)
    print(f"Processing {total_pages} pages...")

    lines_buffer = []

    # 1. Extract and Clean Text
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        raw_lines = text.split('\n')
        for line in raw_lines:
            cleaned = clean_line(line)
            if cleaned:
                lines_buffer.append(cleaned)
        
        if (i + 1) % 100 == 0:
            print(f"Extracted page {i+1}...")

    print(f"Total lines extracted: {len(lines_buffer)}")

    # 2. Structure Parsing
    output_lines = []
    output_lines.append("# Income Tax Act 1961")
    output_lines.append(f"> Source: {os.path.basename(PDF_PATH)}")
    output_lines.append("")

    current_chapter = None
    
    for line in lines_buffer:
        # Check for Chapter
        chapter_match = RE_CHAPTER.match(line)
        if chapter_match:
            current_chapter = chapter_match.group(1)
            output_lines.append(f"\n## CHAPTER {current_chapter}")
            continue
        
        # Check for Section
        section_match = RE_SECTION.match(line)
        if section_match:
            sec_num = section_match.group(1)
            sec_title = section_match.group(2)
            # Markdown H3 for sections
            output_lines.append(f"\n### Section {sec_num}: {sec_title}")
            continue

        # Check for Omitted lines - optional: we can skip them or mark them
        # User said "unnecessary data", so skipping [Omitted] lines that have no content is good.
        # However, keeping a record that it existed is sometimes useful. 
        # Let's keep them but blockquote them to make them distinct, or just write them as text.
        # If the line is JUST "[...Omitted...]" we might want to suppress it if it's very noisy.
        # For now, we write it as normal text but ensure it doesn't break flow.

        # Normal text
        # If it looks like a bullet point (a), (i), (1), formatting it as a list item might be nice
        # but Markdown lists can be finicky with multi-paragraph content.
        # We will keep it as text for now, maybe add a newline if it starts with a clear marker.
        
        # Simple heuristic: if line starts with '(', it *might* be a new clause.
        # Adding a double space at end of previous line ensures hard break in some markdown renderers,
        # but standard markdown requires blank line for new paragraph.
        
        if line.startswith("(") and ")" in line[:10]:
            # Likely a clause
            output_lines.append(f"\n{line}")
        else:
            # Continue text
            output_lines.append(line)

    # 3. Write to File
    print(f"Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print("Conversion complete.")

if __name__ == "__main__":
    main()
