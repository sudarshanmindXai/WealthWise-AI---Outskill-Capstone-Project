"""
Semantic Chunker for Income Tax Forms

These are official IT department forms with:
- Form number and rule reference
- Sections and annexures
- Instructions for filling

Philosophy:
- One form = one or more semantic chunks
- Preserve form context (number, purpose, rule reference)
- Extract key sections for precise retrieval
"""

import json
import re
import os
from pypdf import PdfReader

# Configuration
BASE_DIR = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_2/Forms"
OUTPUT_FILE = os.path.join(BASE_DIR, "it_forms_semantic.jsonl")

MAX_CHUNK_SIZE = 4000

# Form definitions
FORMS = [
    {
        'pdf': 'Form 10E.pdf',
        'form_no': '10E',
        'prefix': 'form10e',
        'rule': '21AA',
        'purpose': 'Relief under section 89 for arrears/advance salary',
        'sections': ['89(1)', '192(2A)']
    },
    {
        'pdf': 'form_16.pdf',
        'form_no': '16',
        'prefix': 'form16',
        'rule': '31(1)(a)',
        'purpose': 'TDS Certificate for salary under section 192',
        'sections': ['192', '203', '194P']
    },
    {
        'pdf': 'form_12ba.pdf',
        'form_no': '12BA',
        'prefix': 'form12ba',
        'rule': '26A(2)(b)',
        'purpose': 'Statement of perquisites, fringe benefits and profits in lieu of salary',
        'sections': ['17(2)', '17(3)']
    },
    {
        'pdf': 'Form 10-IEA.pdf',
        'form_no': '10-IEA',
        'prefix': 'form10iea',
        'rule': '21AGA',
        'purpose': 'Option for new tax regime under section 115BAC',
        'sections': ['115BAC']
    },
]

def clean_text(text):
    """Remove noise from PDF text."""
    # Remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Clean up whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"
    return clean_text(full_text)

def parse_form(pdf_path, form_info):
    """Parse a form PDF into semantic sections."""
    sections = []
    
    full_text = extract_text_from_pdf(pdf_path)
    
    # Split by annexures or parts if present
    parts = re.split(r'(?=\n(?:ANNEXURE|PART\s+[A-Z]|Annexure\s+[IVX]+))', full_text, flags=re.IGNORECASE)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) < 50:
            continue
        
        # Determine section name
        section_match = re.match(r'^(ANNEXURE\s*[IVX]*|PART\s+[A-Z]|Annexure\s+[IVX]+)', part, re.IGNORECASE)
        if section_match:
            section_name = section_match.group(1).strip()
        else:
            section_name = "Main Form" if i == 0 else f"Section {i}"
        
        sections.append({
            'section_name': section_name,
            'content': part
        })
    
    return sections

def chunk_form_section(section, form_info):
    """Convert a form section into one or more chunks."""
    chunks = []
    
    section_name = section['section_name']
    content = section['content']
    
    # Build context
    context = f"Form No. {form_info['form_no']}\n"
    context += f"[See Rule {form_info['rule']}]\n"
    context += f"Purpose: {form_info['purpose']}\n"
    if section_name != "Main Form":
        context += f"Section: {section_name}\n"
    context += "\n"
    
    full_text = context + content
    
    if len(full_text) <= MAX_CHUNK_SIZE:
        chunks.append({
            'section_name': section_name,
            'text': full_text,
            'part': None
        })
    else:
        # Split large sections at paragraph boundaries
        paragraphs = re.split(r'\n\n+', content)
        
        current_chunk = []
        current_len = 0
        part_num = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_len + len(para) > MAX_CHUNK_SIZE - len(context) and current_chunk:
                chunk_text = context + '\n\n'.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append({
                        'section_name': section_name,
                        'text': chunk_text,
                        'part': part_num
                    })
                    part_num += 1
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)
        
        if current_chunk:
            chunk_text = context + '\n\n'.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append({
                    'section_name': section_name,
                    'text': chunk_text,
                    'part': part_num if part_num > 1 else None
                })
    
    return chunks

def process_form(form_info):
    """Process a single form PDF."""
    pdf_path = os.path.join(BASE_DIR, form_info['pdf'])
    
    print(f"\nProcessing Form {form_info['form_no']}...")
    
    if not os.path.exists(pdf_path):
        print(f"  Error: {pdf_path} not found")
        return []
    
    sections = parse_form(pdf_path, form_info)
    print(f"  Found {len(sections)} sections")
    
    all_chunks = []
    seen_doc_ids = {}
    
    for section in sections:
        chunks = chunk_form_section(section, form_info)
        
        for chunk in chunks:
            part_suffix = f"_part{chunk['part']}" if chunk['part'] else ""
            section_slug = re.sub(r'[\W]+', '_', chunk['section_name']).lower()[:20]
            
            base_doc_id = f"{form_info['prefix']}_{section_slug}{part_suffix}"
            
            # Ensure uniqueness
            doc_id = base_doc_id
            if doc_id in seen_doc_ids:
                seen_doc_ids[doc_id] += 1
                doc_id = f"{base_doc_id}_{seen_doc_ids[doc_id]}"
            else:
                seen_doc_ids[doc_id] = 1
            
            all_chunks.append({
                "doc_id": doc_id,
                "doc_type": "form",
                "form_no": form_info['form_no'],
                "rule": form_info['rule'],
                "purpose": form_info['purpose'],
                "section": chunk['section_name'],
                "section_refs": form_info['sections'],
                "text": chunk['text']
            })
    
    return all_chunks

def main():
    print("=== Semantic Chunking for Income Tax Forms ===")
    
    all_chunks = []
    
    for form_info in FORMS:
        chunks = process_form(form_info)
        all_chunks.extend(chunks)
        if chunks:
            sizes = [len(c['text']) for c in chunks]
            print(f"  Created {len(chunks)} chunks (sizes: {min(sizes)} - {max(sizes)} chars)")
    
    # Write output
    print(f"\nWriting {len(all_chunks)} total chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # Summary
    print("\n=== Summary ===")
    for form_info in FORMS:
        form_chunks = [c for c in all_chunks if c['form_no'] == form_info['form_no']]
        print(f"Form {form_info['form_no']}: {len(form_chunks)} chunks")
    print(f"\nTotal: {len(all_chunks)} chunks")
    print("Done!")

if __name__ == "__main__":
    main()
