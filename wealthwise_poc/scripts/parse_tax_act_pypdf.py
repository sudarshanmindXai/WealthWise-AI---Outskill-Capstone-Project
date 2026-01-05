import re
import json
import sys
from pypdf import PdfReader

# Configuration
PDF_PATH = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income-tax-act-1961-as-amended-by-finance-act-2025.pdf"
OUTPUT_PATH = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/chunks.jsonl"
DEBUG_LIMIT = None # Set to an integer (e.g., 200) to limit pages for testing

# Regex Patterns
# Header/Footer cleanup
RE_HEADER = re.compile(r"INCOME-TAX ACT, 1961 - 2024", re.IGNORECASE)
RE_PAGE_NUM = re.compile(r"^\s*\d+\s+of\s+915.*", re.IGNORECASE) # "18 of 915 15/5/2025..."
RE_PAGE_NUM_SIMPLE = re.compile(r"^\s*\d+\s*$", re.IGNORECASE)

# Structure identification
# Section: "10.", "10BA." at start of line
RE_SECTION = re.compile(r"^(\d+[A-Z]*)\.\s") 
# Sub-section: "(1)", "(a)", "(ii)", "(47A)" at start of line
# Note: This is tricky. We need to avoid false positives in text. 
# Usually starts with `(`, chars, `)`, then space/content.
RE_SUBSECTION = re.compile(r"^\(([\w]+)\)\s")

def clean_line(line):
    """Removes headers, footers, and extra whitespace."""
    if RE_HEADER.search(line):
        return None
    if RE_PAGE_NUM.search(line):
        return None
    if RE_PAGE_NUM_SIMPLE.match(line): # Just a number line
        return None
    return line.strip()

def slugify(text):
    """Creates a basic slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')

def create_chunk(section, subsection_stack, text):
    """Builds the JSON object."""
    if not text.strip():
        return None
        
    s_section = section if section else "preamble"
    
    # Construct subsection string from stack e.g., "(1)(a)"
    # stack elements are just "1", "a". We wrap them back.
    if subsection_stack:
        s_subsection = "".join([f"({x})" for x in subsection_stack])
    else:
        s_subsection = None
        
    # ID Construction
    # it_act_10ba_1_a
    # If no subsection, just it_act_10ba
    sub_slug = "_" + "_".join([slugify(x) for x in subsection_stack]) if subsection_stack else ""
    doc_id = f"it_act_{slugify(s_section)}{sub_slug}"
    
    return {
        "doc_id": doc_id,
        "doc_type": "act",
        "section": s_section,
        "sub_section": s_subsection,
        "ay": ["2025-26"],
        "text": text.strip()
    }

def main():
    reader = PdfReader(PDF_PATH)
    total_pages = len(reader.pages)
    print(f"Processing {total_pages} pages...")
    
    output_file = open(OUTPUT_PATH, "w", encoding="utf-8")
    
    current_section = None
    # Stack to track hierarchy. Ideally simple list of markers like ['1', 'a']
    # But flattening PDF hierarchy is hard because indentation isn't reliable.
    # We will try a "latest seen" approach for now.
    # Actually, the user asked for "One logical law chunk". 
    # If we have nested structure:
    # (1) Blah...
    #   (a) Blah sub...
    # We should probably treat (1) as a chunk (intro text) and (a) as separate chunk?
    # Or combine them?
    # RAG usually benefits from atomic chunks.
    # We will treat every detected "marker" as starting a new chunk.
    
    current_subsection_stack = [] 
    current_text_buffer = []
    
    # We need a way to detect if a subsection is a child of the previous one or a sibling.
    # E.g. (1) -> (2) is sibling. (1) -> (a) is child.
    # Tax act usually goes: Section -> (1) -> (i) -> (a) OR Section -> (a)
    # Common hierarchy: Section -> Subsection (numeric) -> Clause (alpha) -> Sub-clause (roman)
    # But it varies.
    # SIMPLIFICATION: We will store the *last* fully identified marker as the "sub_section" string
    # BUT the user example showed "(1A)".
    # Let's try to maintain a stack based on common sense pattern matching or just flatten?
    # Given the complexity and "rule-based" nature requested (regex),
    # verifying strict hierarchy without visual layout is hard.
    # Strategy: 
    # If we hit a Section -> Clear stack. 
    # If we hit a sub-section ->
    #   If it looks like a new level (e.g. going from 1 to a), push?
    #   Or just keep it simple: The "sub_section" field is the label of the *current* atomic block.
    #   The 'stack' might be overkill if we just want retrieval.
    #   Let's stick to the User Example: "sub_section": "(1A)". 
    #   We will just capture the IMMEDIATE marker that started this block.
    #   However, context is useful. "clase (a) of sub-section (1)".
    #   Revised Strategy:
    #   Just capture the *immediately preceding marker* as the `sub_section`.
    #   AND accumulate text until the next marker.
    
    last_marker = None # e.g. "(1)" or "(a)"

    iterator = range(total_pages)
    if DEBUG_LIMIT:
        iterator = range(min(total_pages, DEBUG_LIMIT))

    for i in iterator:
        page = reader.pages[i]
        text = page.extract_text()
        if not text:
            continue
            
        lines = text.split('\n')
        
        for line in lines:
            cleaned = clean_line(line)
            if cleaned is None:
                continue
            
            # Check for Section Start (High priority)
            sec_match = RE_SECTION.match(cleaned)
            if sec_match:
                # Flush previous
                if current_text_buffer:
                    chunk = create_chunk(current_section, [last_marker] if last_marker else [], " ".join(current_text_buffer))
                    if chunk:
                        output_file.write(json.dumps(chunk) + "\n")
                
                # Start new Section
                current_section = sec_match.group(1) # "10"
                last_marker = None # Reset subsection
                current_text_buffer = [cleaned[sec_match.end():].strip()] # Start text after "10. "
                continue
                
            # Check for Sub-section Start
            # Only if we are inside a section
            if current_section:
                sub_match = RE_SUBSECTION.match(cleaned)
                if sub_match:
                    # Flush previous chunk (which was the content of the PREVIOUS marker)
                    # Note: This splits "Section 10... text..." into its own chunk before "(1)" starts.
                    if current_text_buffer:
                        # If the buffer is just empty or tiny, maybe don't flush? 
                        # But legal text usually has preamble.
                        chunk = create_chunk(current_section, [last_marker] if last_marker else [], " ".join(current_text_buffer))
                        if chunk:
                            output_file.write(json.dumps(chunk) + "\n")
                    
                    # Start new Sub-section
                    # raw_marker = sub_match.group(0).strip() # "(1)"
                    inner_marker = sub_match.group(1) # "1"
                    last_marker = inner_marker # Store just "1" or "a". Logic will wrap it in ()
                    
                    current_text_buffer = [cleaned[sub_match.end():].strip()]
                    continue
            
            # Normal text line
            current_text_buffer.append(cleaned)
            
    # Flush final buffer
    if current_text_buffer and current_section:
        chunk = create_chunk(current_section, [last_marker] if last_marker else [], " ".join(current_text_buffer))
        if chunk:
            output_file.write(json.dumps(chunk) + "\n")
            
    output_file.close()
    print(f"Done! Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
