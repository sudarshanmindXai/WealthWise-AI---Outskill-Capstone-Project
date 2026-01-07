"""
Semantic Chunker for Income Tax Educational Booklets

These are educational/instructional documents with:
- Headings and subheadings
- Explanatory paragraphs
- Examples and illustrations
- Tables

Philosophy:
- One logical section = one chunk
- Preserve headings for context
- Keep examples with their explanations
"""

import json
import re
import os
from pypdf import PdfReader

MAX_CHUNK_SIZE = 5000

# Patterns
RE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
RE_HEADING = re.compile(r"^([A-Z][A-Z\s\-]+)$", re.MULTILINE)

def slugify(text):
    """Create a clean slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')[:50]

def clean_text(text):
    """Remove noise from PDF text."""
    text = RE_PAGE_NUMBER.sub('', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove "[As amended by Finance Act, 2025]" headers
    text = re.sub(r'\[As amended by Finance Act[^\]]*\]', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_title_from_first_page(text):
    """Extract document title from first page."""
    # Look for title patterns
    patterns = [
        r'TAX ON (SHORT[- ]TERM CAPITAL GAINS)',
        r'TAX ON (LONG[- ]TERM CAPITAL GAINS)',
        r'(PRESUMPTIVE\s+TAXATION)',
        r'Instructions.*?(ITR-\d+)',
        r'^([A-Z][A-Z\s\-]{10,50})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Tax Guide"

def parse_booklet(pdf_path, doc_prefix):
    """Parse a booklet PDF into sections."""
    reader = PdfReader(pdf_path)
    
    # Extract all text
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n\n"
    
    full_text = clean_text(full_text)
    
    # Get title from first page
    first_page = reader.pages[0].extract_text()
    title = extract_title_from_first_page(first_page)
    
    # Parse into sections based on headings
    sections = []
    
    # Split by major heading patterns
    # Look for lines that are ALL CAPS (headings)
    lines = full_text.split('\n')
    
    current_heading = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this is a heading (ALL CAPS, reasonable length)
        if (stripped and 
            len(stripped) > 5 and 
            len(stripped) < 100 and
            stripped.isupper() and
            not re.match(r'^[IVXLC]+\.?\s*$', stripped)):  # Not just roman numerals
            
            # Save previous section
            if current_content:
                content = '\n'.join(current_content)
                if len(content.strip()) > 100:
                    sections.append({
                        'heading': current_heading or title,
                        'content': content,
                        'title': title
                    })
            
            current_heading = stripped.title()  # Convert to Title Case
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        content = '\n'.join(current_content)
        if len(content.strip()) > 100:
            sections.append({
                'heading': current_heading or title,
                'content': content,
                'title': title
            })
    
    # If no sections found, treat whole document as one section
    if not sections:
        sections.append({
            'heading': title,
            'content': full_text,
            'title': title
        })
    
    return sections, title

def chunk_section(section, doc_prefix):
    """Convert a section into one or more chunks."""
    chunks = []
    
    heading = section['heading']
    content = section['content'].strip()
    title = section['title']
    
    if len(content) < 50:
        return []
    
    # Build context prefix
    context = f"{title}\n"
    if heading and heading != title:
        context += f"Section: {heading}\n"
    context += "\n"
    
    full_text = context + content
    
    if len(full_text) <= MAX_CHUNK_SIZE:
        chunks.append({
            'heading': heading,
            'title': title,
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
                        'heading': heading,
                        'title': title,
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
                    'heading': heading,
                    'title': title,
                    'text': chunk_text,
                    'part': part_num if part_num > 1 else None
                })
    
    return chunks

def process_booklet(pdf_path, output_path, doc_prefix, doc_name):
    """Process a single booklet and write to JSONL."""
    print(f"\nProcessing {doc_name}...")
    
    if not os.path.exists(pdf_path):
        print(f"  Error: {pdf_path} not found")
        return 0
    
    sections, title = parse_booklet(pdf_path, doc_prefix)
    print(f"  Found {len(sections)} sections")
    print(f"  Title: {title}")
    
    # Convert to chunks
    all_chunks = []
    seen_doc_ids = {}
    section_num = 0
    
    for section in sections:
        section_num += 1
        chunks = chunk_section(section, doc_prefix)
        
        for chunk in chunks:
            part_suffix = f"_part{chunk['part']}" if chunk['part'] else ""
            heading_slug = slugify(chunk['heading'])[:30] if chunk['heading'] else ""
            
            base_doc_id = f"{doc_prefix}_sec{section_num}"
            if heading_slug:
                base_doc_id += f"_{heading_slug}"
            base_doc_id += part_suffix
            
            # Ensure uniqueness
            doc_id = base_doc_id
            if doc_id in seen_doc_ids:
                seen_doc_ids[doc_id] += 1
                doc_id = f"{base_doc_id}_{seen_doc_ids[doc_id]}"
            else:
                seen_doc_ids[doc_id] = 1
            
            all_chunks.append({
                "doc_id": doc_id,
                "doc_type": "booklet",
                "booklet": doc_name,
                "title": chunk['title'],
                "section": chunk['heading'],
                "ay": ["2025-26"],
                "text": chunk['text']
            })
    
    # Write output
    print(f"  Writing {len(all_chunks)} chunks to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # Stats
    if all_chunks:
        sizes = [len(c['text']) for c in all_chunks]
        print(f"  Chunk sizes: {min(sizes)} - {max(sizes)} chars (avg: {sum(sizes)//len(sizes)})")
    
    return len(all_chunks)

def main():
    base_dir = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_2/side_hustle_booklets"
    
    booklets = [
        {
            'pdf': f"{base_dir}/14- stcg.pdf",
            'output': f"{base_dir}/stcg_semantic.jsonl",
            'prefix': 'stcg',
            'name': 'Short-Term Capital Gains'
        },
        {
            'pdf': f"{base_dir}/15- ltcg.pdf",
            'output': f"{base_dir}/ltcg_semantic.jsonl",
            'prefix': 'ltcg',
            'name': 'Long-Term Capital Gains'
        },
        {
            'pdf': f"{base_dir}/Instructions_ITR4_AY2021_22.pdf",
            'output': f"{base_dir}/itr4_instructions_semantic.jsonl",
            'prefix': 'itr4',
            'name': 'ITR-4 Instructions'
        },
        {
            'pdf': f"{base_dir}/presumptive-taxation-english.pdf",
            'output': f"{base_dir}/presumptive_tax_semantic.jsonl",
            'prefix': 'presumptive',
            'name': 'Presumptive Taxation'
        },
    ]
    
    total_chunks = 0
    
    print("=== Semantic Chunking for Side Hustle Booklets ===")
    
    for booklet in booklets:
        chunks = process_booklet(
            booklet['pdf'],
            booklet['output'],
            booklet['prefix'],
            booklet['name']
        )
        total_chunks += chunks
    
    print(f"\n=== Summary ===")
    print(f"Total chunks created across all booklets: {total_chunks}")
    print("Done!")

if __name__ == "__main__":
    main()
