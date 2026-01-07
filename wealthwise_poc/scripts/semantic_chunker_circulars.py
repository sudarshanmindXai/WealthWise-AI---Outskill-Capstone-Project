"""
Semantic Chunker for CBDT Circulars

Circulars are guidance documents with:
- Numbered paragraphs (1, 2, 3, etc.)
- Subject headings
- Tables (tax rates, etc.)
- Annexures, Forms

Philosophy:
- One logical section = one chunk
- Preserve context (circular number, subject, date)
- Include tables with their explanatory text
"""

import json
import re
import os
from pypdf import PdfReader

# Configuration
INPUT_PDF = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_2/circular-no-03-2025.pdf"
OUTPUT_JSONL = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_2/circular_03_2025_semantic.jsonl"

MAX_CHUNK_SIZE = 6000

# Patterns
RE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
RE_PARAGRAPH_START = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

def slugify(text):
    """Create a clean slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')[:50]

def clean_text(text):
    """Remove noise from PDF text."""
    text = RE_PAGE_NUMBER.sub('', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_circular_metadata(first_page_text):
    """Extract circular metadata from first page."""
    metadata = {
        'circular_no': None,
        'date': None,
        'subject': None,
        'file_no': None
    }
    
    # File number
    file_match = re.search(r'F\.\s*No\.\s*([^\n]+)', first_page_text)
    if file_match:
        metadata['file_no'] = file_match.group(1).strip()
    
    # Circular number
    circ_match = re.search(r'CIRCULAR\s*NO\s*[:\.]?\s*(\d+/\d+)', first_page_text, re.IGNORECASE)
    if circ_match:
        metadata['circular_no'] = circ_match.group(1)
    
    # Date
    date_match = re.search(r'Dated\s+(?:the\s+)?(\d+(?:st|nd|rd|th)?\s+\w+,?\s*\d{4})', first_page_text, re.IGNORECASE)
    if date_match:
        metadata['date'] = date_match.group(1).strip()
    
    # Subject
    subj_match = re.search(r'SUBJECT\s*[:\-]\s*(.+?)(?:\*{3,}|$)', first_page_text, re.IGNORECASE | re.DOTALL)
    if subj_match:
        subject = subj_match.group(1).strip()
        subject = re.sub(r'\s+', ' ', subject)
        metadata['subject'] = subject[:200]  # Limit length
    
    return metadata

def parse_circular(pdf_path):
    """Parse the circular PDF into sections."""
    reader = PdfReader(pdf_path)
    
    # Extract all text
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"
    
    # Get metadata from first page
    first_page = reader.pages[0].extract_text()
    metadata = extract_circular_metadata(first_page)
    
    # Parse into sections based on numbered paragraphs
    sections = []
    
    # Split by lines and look for paragraph starts
    lines = full_text.split('\n')
    current_para_num = None
    current_content = []
    
    for line in lines:
        # Check if this line starts with a paragraph number
        # Pattern: "N. " at start of line, where N is 1-99
        para_match = re.match(r'^(\d{1,2})\.\s+([A-Z])', line)
        
        if para_match:
            para_num = para_match.group(1)
            
            # Skip if it looks like a year (1961, 2024, etc.)
            if int(para_num) > 100:
                current_content.append(line)
                continue
            
            # Save previous section
            if current_para_num is not None:
                content = '\n'.join(current_content)
                if len(content.strip()) > 50:
                    section_type = "paragraph"
                    if "Form No." in content or "Annexure" in content:
                        section_type = "form"
                    elif re.search(r'Rs\.\s*\d+|per\s+cent', content):
                        section_type = "table"
                    
                    sections.append({
                        'para_num': current_para_num,
                        'section_type': section_type,
                        'content': content,
                        'metadata': metadata
                    })
            elif current_content:
                # Intro section
                content = '\n'.join(current_content)
                if len(content.strip()) > 50:
                    sections.append({
                        'para_num': '0',
                        'section_type': 'intro',
                        'content': content,
                        'metadata': metadata
                    })
            
            current_para_num = para_num
            current_content = [line]
        else:
            current_content.append(line)
    
    # Save last section
    if current_para_num is not None and current_content:
        content = '\n'.join(current_content)
        if len(content.strip()) > 50:
            section_type = "paragraph"
            if "Form No." in content or "Annexure" in content:
                section_type = "form"
            
            sections.append({
                'para_num': current_para_num,
                'section_type': section_type,
                'content': content,
                'metadata': metadata
            })
    
    return sections, metadata

def chunk_section(section, metadata):
    """Convert a section into one or more chunks."""
    chunks = []
    
    para_num = section['para_num']
    section_type = section['section_type']
    content = clean_text(section['content'])
    
    if len(content) < 50:
        return []
    
    # Build context prefix
    circular_no = metadata.get('circular_no', 'No. 03/2025')
    subject = metadata.get('subject', 'TDS on Salaries')
    date = metadata.get('date', '')
    
    context = f"Circular {circular_no}"
    if date:
        context += f" dated {date}"
    context += "\n"
    if para_num != '0':
        context += f"Paragraph {para_num}\n"
    context += "\n"
    
    full_text = context + content
    
    if len(full_text) <= MAX_CHUNK_SIZE:
        chunks.append({
            'para_num': para_num,
            'section_type': section_type,
            'text': full_text,
            'part': None
        })
    else:
        # Split large sections
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.;])\s+(?=[A-Z(])', content)
        
        current_chunk = []
        current_len = 0
        part_num = 1
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if current_len + len(sent) > MAX_CHUNK_SIZE - len(context) and current_chunk:
                chunk_text = context + ' '.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append({
                        'para_num': para_num,
                        'section_type': section_type,
                        'text': chunk_text,
                        'part': part_num
                    })
                    part_num += 1
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent)
        
        if current_chunk:
            chunk_text = context + ' '.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append({
                    'para_num': para_num,
                    'section_type': section_type,
                    'text': chunk_text,
                    'part': part_num if part_num > 1 else None
                })
    
    return chunks

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: {INPUT_PDF} not found")
        return

    print(f"Parsing circular from {INPUT_PDF}...")
    sections, metadata = parse_circular(INPUT_PDF)
    print(f"Found {len(sections)} sections")
    print(f"Circular: {metadata.get('circular_no')}")
    print(f"Subject: {metadata.get('subject', '')[:80]}...")
    
    # Convert to chunks
    all_chunks = []
    seen_doc_ids = {}
    
    for section in sections:
        chunks = chunk_section(section, metadata)
        for chunk in chunks:
            part_suffix = f"_part{chunk['part']}" if chunk['part'] else ""
            
            base_doc_id = f"circular_03_2025_para{chunk['para_num']}{part_suffix}"
            
            # Ensure uniqueness
            doc_id = base_doc_id
            if doc_id in seen_doc_ids:
                seen_doc_ids[doc_id] += 1
                doc_id = f"{base_doc_id}_{seen_doc_ids[doc_id]}"
            else:
                seen_doc_ids[doc_id] = 1
            
            all_chunks.append({
                "doc_id": doc_id,
                "doc_type": "circular",
                "circular_no": metadata.get('circular_no', '03/2025'),
                "date": metadata.get('date', ''),
                "subject": metadata.get('subject', ''),
                "paragraph": chunk['para_num'],
                "section_type": chunk['section_type'],
                "ay": ["2025-26"],
                "text": chunk['text']
            })

    # Write output
    print(f"\nWriting {len(all_chunks)} semantic chunks to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    # Stats
    print("\n=== Semantic Chunking Stats ===")
    print(f"Total sections parsed: {len(sections)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    if all_chunks:
        sizes = [len(c['text']) for c in all_chunks]
        print(f"Min chunk size: {min(sizes)} chars")
        print(f"Max chunk size: {max(sizes)} chars")
        print(f"Avg chunk size: {sum(sizes)//len(sizes)} chars")
        
        # Count by section type
        types = {}
        for c in all_chunks:
            t = c['section_type']
            types[t] = types.get(t, 0) + 1
        print(f"\nChunks by type:")
        for t, count in sorted(types.items()):
            print(f"  {t}: {count}")
    
    # Sample output
    print("\n=== Sample Chunks ===")
    for chunk in all_chunks[:3]:
        print(f"  {chunk['doc_id']}: {chunk['text'][:100]}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
