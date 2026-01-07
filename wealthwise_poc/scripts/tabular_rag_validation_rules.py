"""
Tabular RAG Chunker for ITR Validation Rules

These are structured table-based documents with:
- Numbered validation rules
- Categories (A=Block, B=Warning, D=Advisory)
- References to schedules and sections

Philosophy:
- One rule = one chunk (row-level chunking)
- Extract metadata: category, schedule, section references
- Enable precise retrieval for form validation queries
"""

import json
import re
import os
from pypdf import PdfReader

# Configuration
BASE_DIR = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_2/Forms/Validation rules"
OUTPUT_DIR = BASE_DIR

# ITR validation rule files
VALIDATION_FILES = [
    {
        'pdf': 'CBDT_e-Filing_ITR 1_Validation Rules_AY 2025-26_V1.1.pdf',
        'itr_form': 'ITR-1',
        'prefix': 'itr1',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT__e-Filing_ITR 2_Validation Rules_AY 2025-26_V1.0.pdf',
        'itr_form': 'ITR-2',
        'prefix': 'itr2',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT_e-filing_ITR-3_Validation Rules_V1.0_AY 25-26.pdf',
        'itr_form': 'ITR-3',
        'prefix': 'itr3',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT_e-Filing_ITR 4_Validation Rules_AY 2025-26_V1.1.pdf',
        'itr_form': 'ITR-4',
        'prefix': 'itr4',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT_e-Filing_ITR 5_Validation Rules_V 1.0.pdf',
        'itr_form': 'ITR-5',
        'prefix': 'itr5',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT__e-Filing_ITR-6_Validation Rules_Version 1.0 (1).pdf',
        'itr_form': 'ITR-6',
        'prefix': 'itr6',
        'ay': '2025-26'
    },
    {
        'pdf': 'CBDT_e-Filing_ITR-7_Validation Rules_V 1.0_AY 25-26.pdf',
        'itr_form': 'ITR-7',
        'prefix': 'itr7',
        'ay': '2025-26'
    },
]

# Patterns for extraction
RE_RULE_NUMBER = re.compile(r'^(\d+)\.\s+', re.MULTILINE)
RE_SECTION_REF = re.compile(r'(?:u/s|section|sec\.|Sec)\s*(\d+[A-Z]*(?:\([^)]+\))?)', re.IGNORECASE)
RE_SCHEDULE_REF = re.compile(r'(?:Schedule|Sch\.?)\s+"?([^"]+?)"?(?:\s|,|$)', re.IGNORECASE)

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def extract_section_refs(text):
    """Extract section references from rule text."""
    matches = RE_SECTION_REF.findall(text)
    return list(set(matches)) if matches else []

def extract_schedule_refs(text):
    """Extract schedule references from rule text."""
    matches = RE_SCHEDULE_REF.findall(text)
    # Clean up matches
    cleaned = []
    for m in matches:
        m = m.strip().rstrip('.,;')
        if m and len(m) < 50:  # Avoid capturing too much
            cleaned.append(m)
    return list(set(cleaned)) if cleaned else []

def determine_category(text, current_category):
    """Determine rule category from context."""
    # Look for category headers
    if 'Category A:' in text or 'Category A Rules' in text:
        return 'A'
    elif 'Category B:' in text or 'Category B Rules' in text:
        return 'B'
    elif 'Category D:' in text or 'Category D Rules' in text:
        return 'D'
    return current_category

def parse_validation_rules(pdf_path, itr_form, prefix, ay):
    """Parse validation rules from PDF into row-level chunks."""
    rules = []
    
    full_text = extract_text_from_pdf(pdf_path)
    
    # Split into lines and process
    lines = full_text.split('\n')
    
    current_category = 'A'  # Default to Category A
    current_rule_num = None
    current_rule_lines = []
    last_seen_rule_num = 0  # Track sequential rule numbers
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and page numbers
        if not line or re.match(r'^Page\s*\d+', line):
            continue
        
        # Check for category changes
        if re.search(r'Category\s+A[:\s]|Category\s+A\s+Rules', line, re.IGNORECASE):
            current_category = 'A'
            continue
        elif re.search(r'Category\s+B[:\s]|Category\s+B\s+Rules', line, re.IGNORECASE):
            current_category = 'B'
            continue
        elif re.search(r'Category\s+D[:\s]|Category\s+D\s+Rules', line, re.IGNORECASE):
            current_category = 'D'
            continue
        
        # Check for new rule start - must be at beginning of line
        rule_match = re.match(r'^(\d+)\.\s+(.+)', line)
        if rule_match:
            rule_num = int(rule_match.group(1))
            rule_text_start = rule_match.group(2)
            
            # Validate this is a real rule number (should be sequential or close)
            # Skip if it looks like a year (1961, 2024) or random number
            if rule_num > 1000 or rule_num < 1:
                if current_rule_num:
                    current_rule_lines.append(line)
                continue
            
            # Check if this could be a continuation of numbering
            # Allow for some gaps but not huge jumps
            if rule_num > last_seen_rule_num + 50 and last_seen_rule_num > 0:
                if current_rule_num:
                    current_rule_lines.append(line)
                continue
            
            # Save previous rule
            if current_rule_num and current_rule_lines:
                rule_text = ' '.join(current_rule_lines)
                # Limit rule text size to 3000 chars
                if len(rule_text) > 3000:
                    rule_text = rule_text[:3000] + "..."
                
                if len(rule_text) > 20:
                    section_refs = extract_section_refs(rule_text)
                    schedule_refs = extract_schedule_refs(rule_text)
                    
                    rules.append({
                        'rule_number': str(current_rule_num),
                        'category': current_category,
                        'rule_text': rule_text,
                        'section_refs': section_refs,
                        'schedule_refs': schedule_refs,
                        'itr_form': itr_form,
                        'ay': ay
                    })
            
            current_rule_num = rule_num
            last_seen_rule_num = rule_num
            current_rule_lines = [f"{rule_num}. {rule_text_start}"]
        else:
            # Continue accumulating text for current rule
            if current_rule_num and line:
                # Only add if we haven't accumulated too much
                current_text = ' '.join(current_rule_lines)
                if len(current_text) < 3000:
                    current_rule_lines.append(line)
    
    # Save last rule
    if current_rule_num and current_rule_lines:
        rule_text = ' '.join(current_rule_lines)
        if len(rule_text) > 3000:
            rule_text = rule_text[:3000] + "..."
        
        if len(rule_text) > 20:
            section_refs = extract_section_refs(rule_text)
            schedule_refs = extract_schedule_refs(rule_text)
            
            rules.append({
                'rule_number': str(current_rule_num),
                'category': current_category,
                'rule_text': rule_text,
                'section_refs': section_refs,
                'schedule_refs': schedule_refs,
                'itr_form': itr_form,
                'ay': ay
            })
    
    return rules

def process_validation_file(file_info):
    """Process a single validation rules PDF."""
    pdf_path = os.path.join(BASE_DIR, file_info['pdf'])
    output_path = os.path.join(OUTPUT_DIR, f"{file_info['prefix']}_validation_rules.jsonl")
    
    print(f"\nProcessing {file_info['itr_form']}...")
    
    if not os.path.exists(pdf_path):
        print(f"  Error: {pdf_path} not found")
        return 0
    
    rules = parse_validation_rules(
        pdf_path,
        file_info['itr_form'],
        file_info['prefix'],
        file_info['ay']
    )
    
    print(f"  Found {len(rules)} rules")
    
    # Convert to JSONL format
    chunks = []
    seen_doc_ids = {}
    
    for rule in rules:
        base_doc_id = f"{file_info['prefix']}_rule_{rule['rule_number']}"
        
        # Ensure uniqueness
        doc_id = base_doc_id
        if doc_id in seen_doc_ids:
            seen_doc_ids[doc_id] += 1
            doc_id = f"{base_doc_id}_{seen_doc_ids[doc_id]}"
        else:
            seen_doc_ids[doc_id] = 1
        
        # Build readable text
        text = f"{file_info['itr_form']} Validation Rule #{rule['rule_number']}\n"
        text += f"Category: {rule['category']} "
        if rule['category'] == 'A':
            text += "(Blocking - Return will not be uploaded)\n"
        elif rule['category'] == 'B':
            text += "(Warning - Possible defect u/s 139(9))\n"
        elif rule['category'] == 'D':
            text += "(Advisory - Claim may not be allowed)\n"
        text += f"\n{rule['rule_text']}"
        
        chunks.append({
            "doc_id": doc_id,
            "doc_type": "validation_rule",
            "itr_form": file_info['itr_form'],
            "category": rule['category'],
            "rule_number": rule['rule_number'],
            "section_refs": rule['section_refs'],
            "schedule_refs": rule['schedule_refs'],
            "ay": file_info['ay'],
            "text": text
        })
    
    # Write output
    print(f"  Writing {len(chunks)} chunks to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # Stats
    if chunks:
        sizes = [len(c['text']) for c in chunks]
        print(f"  Chunk sizes: {min(sizes)} - {max(sizes)} chars (avg: {sum(sizes)//len(sizes)})")
        
        # Category breakdown
        categories = {}
        for c in chunks:
            cat = c['category']
            categories[cat] = categories.get(cat, 0) + 1
        print(f"  Categories: {dict(categories)}")
    
    return len(chunks)

def main():
    print("=== Tabular RAG for ITR Validation Rules ===")
    
    total_chunks = 0
    
    for file_info in VALIDATION_FILES:
        chunks = process_validation_file(file_info)
        total_chunks += chunks
    
    print(f"\n=== Summary ===")
    print(f"Total validation rules processed: {total_chunks}")
    print("Done!")

if __name__ == "__main__":
    main()
