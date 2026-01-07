"""
Semantic Chunker for Finance Bill 2024

Structure:
- Chapters (Chapter I, II, III, etc.)
- Clauses (numbered 1-157+) - each clause amends a section of the IT Act
- Sub-clauses (a), (b), (i), (ii), etc.

Philosophy:
- One clause = one chunk (a complete amendment)
- Clean headers, page numbers, and noise
- doc_id reflects the clause and what it amends
"""

import json
import re
import os
from pypdf import PdfReader

# Configuration
INPUT_PDF = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/Finance_Bill.pdf"
OUTPUT_JSONL = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/finance_bill_semantic.jsonl"

MAX_CHUNK_SIZE = 6000

# Patterns
RE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
RE_MARGIN_REF = re.compile(r"\n\s*\d+\s+of\s+\d+\.\s*", re.MULTILINE)
RE_AMENDMENT_OF = re.compile(r"Amendment of section\s+(\d+[A-Z]*)", re.IGNORECASE)
RE_SUBSTITUTION = re.compile(r"Substitution of.*?section[s]?\s+(\d+[A-Z]*)", re.IGNORECASE)
RE_INSERTION = re.compile(r"Insertion of.*?section\s+(\d+[A-Z]*)", re.IGNORECASE)
RE_OMISSION = re.compile(r"Omission of section\s+(\d+[A-Z]*)", re.IGNORECASE)

# Chapter info (actual pages, 0-indexed) - from analysis
CHAPTERS = [
    {'num': 'I', 'title': 'PRELIMINARY', 'start_page': 8, 'end_page': 9},  # Page 9
    {'num': 'II', 'title': 'RATES OF INCOME-TAX', 'start_page': 8, 'end_page': 32},  # Pages 9-32
    {'num': 'III', 'title': 'DIRECT TAXES', 'start_page': 32, 'end_page': 79},  # Pages 33-79
    {'num': 'IV', 'title': 'THE DIRECT TAX VIVAD SE VISHWAS SCHEME, 2024', 'start_page': 79, 'end_page': 91},  # Pages 80-91
    {'num': 'V', 'title': 'INDIRECT TAXES', 'start_page': 91, 'end_page': 111},  # Pages 92-111
    {'num': 'VI', 'title': 'MISCELLANEOUS', 'start_page': 111, 'end_page': 225},  # Pages 112-end
]

def slugify(text):
    """Create a clean slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')[:60]

def clean_text(text):
    """Remove noise from PDF text."""
    text = RE_PAGE_NUMBER.sub('', text)
    text = RE_MARGIN_REF.sub(' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_section_being_amended(title):
    """Extract the section number being amended from clause title."""
    if not title:
        return None
    patterns = [RE_AMENDMENT_OF, RE_SUBSTITUTION, RE_INSERTION, RE_OMISSION]
    for pattern in patterns:
        match = pattern.search(title)
        if match:
            return match.group(1)
    return None

def parse_finance_bill(pdf_path):
    """Parse the Finance Bill PDF into clauses."""
    reader = PdfReader(pdf_path)
    all_clauses = []
    
    for chapter in CHAPTERS:
        ch_num = chapter['num']
        ch_title = chapter['title']
        start_page = chapter['start_page']
        end_page = min(chapter['end_page'], len(reader.pages))
        
        # Extract text for this chapter
        chapter_text = ""
        for page_num in range(start_page, end_page):
            chapter_text += reader.pages[page_num].extract_text() + "\n"
        
        # Parse clauses from chapter text
        clauses = parse_clauses_from_text(chapter_text, ch_num, ch_title)
        all_clauses.extend(clauses)
    
    return all_clauses

def parse_clauses_from_text(text, chapter_num, chapter_title):
    """Parse clauses from chapter text."""
    clauses = []
    lines = text.split('\n')
    
    current_clause_num = None
    current_clause_title = None
    current_clause_lines = []
    pending_title = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty and page number lines
        if not line or re.match(r'^\d+\s*$', line):
            i += 1
            continue
        
        # Skip chapter headers in content
        if re.match(r'^CHAPTER\s+[IVXLC]+', line):
            i += 1
            continue
        
        # Pattern 1: Title + clause on same line
        # e.g., "Amendment of\nsection 11. 5. In section 11..."
        # This appears as "section 11. 5. In section 11..." on a line
        combined_match = re.match(r'^section\s+(\d+[A-Z]*)\.\s*(\d+)\.\s+(.+)', line, re.IGNORECASE)
        if combined_match and pending_title:
            section_num = combined_match.group(1)
            clause_num = combined_match.group(2)
            clause_content = combined_match.group(3)
            
            # Save previous clause
            if current_clause_num and current_clause_lines:
                clause_text = '\n'.join(current_clause_lines)
                if len(clause_text) > 100:
                    clauses.append({
                        'chapter': chapter_num,
                        'chapter_title': chapter_title,
                        'clause_num': current_clause_num,
                        'clause_title': current_clause_title,
                        'content': clause_text
                    })
            
            # Build title from pending + section
            full_title = pending_title + " section " + section_num
            current_clause_num = clause_num
            current_clause_title = full_title
            current_clause_lines = [f"{clause_num}. {clause_content}"]
            pending_title = None
            i += 1
            continue
        
        # Pattern 2: Just "section X. N. Content" without prior pending title
        combined_match2 = re.match(r'^section\s+(\d+[A-Z]*)\.\s*(\d+)\.\s+(.+)', line, re.IGNORECASE)
        if combined_match2:
            section_num = combined_match2.group(1)
            clause_num = combined_match2.group(2)
            clause_content = combined_match2.group(3)
            
            # Save previous clause
            if current_clause_num and current_clause_lines:
                clause_text = '\n'.join(current_clause_lines)
                if len(clause_text) > 100:
                    clauses.append({
                        'chapter': chapter_num,
                        'chapter_title': chapter_title,
                        'clause_num': current_clause_num,
                        'clause_title': current_clause_title,
                        'content': clause_text
                    })
            
            current_clause_num = clause_num
            current_clause_title = f"Amendment of section {section_num}"
            current_clause_lines = [f"{clause_num}. {clause_content}"]
            pending_title = None
            i += 1
            continue
        
        # Check for clause title in margin (multi-line)
        if re.match(r'^(Amendment of|Substitution of|Insertion of|Omission of|Short title|Definitions|Income-tax|Filing of|Amount payable|Time and manner|Immunity from|No refund|No benefit|Scheme not|Power to)', line, re.IGNORECASE):
            # This could be followed by section number on same or next line
            pending_title = line
            i += 1
            continue
        
        # Check for clause start: "N. In section X..." or "N. (1) This..."
        clause_match = re.match(r'^(\d+)\.\s+(.+)', line)
        if clause_match:
            clause_num = clause_match.group(1)
            clause_content = clause_match.group(2)
            
            # Skip if it looks like a table row
            invalid_patterns = [
                r'^(From Rs|Above Rs|Up to Rs|Nil|Rs\.|[A-Z]\.\s*$)',
            ]
            is_invalid = any(re.match(p, clause_content, re.IGNORECASE) for p in invalid_patterns)
            
            # Valid if we have a pending title OR content looks like legal text
            valid_starts = ['In ', 'For ', 'The ', 'After ', 'With ', 'This ', 'Subject ', '(1)', '(a)', 'Every ', 'On ', 'Where ', 'Notwithstanding']
            is_valid = pending_title is not None or any(clause_content.startswith(s) for s in valid_starts)
            
            if is_valid and not is_invalid:
                # Save previous clause
                if current_clause_num and current_clause_lines:
                    clause_text = '\n'.join(current_clause_lines)
                    if len(clause_text) > 100:
                        clauses.append({
                            'chapter': chapter_num,
                            'chapter_title': chapter_title,
                            'clause_num': current_clause_num,
                            'clause_title': current_clause_title,
                            'content': clause_text
                        })
                
                current_clause_num = clause_num
                current_clause_title = pending_title if pending_title else f"Clause {clause_num}"
                current_clause_lines = [line]
                pending_title = None
                i += 1
                continue
        
        # Accumulate content for current clause
        if current_clause_num is not None:
            current_clause_lines.append(line)
        
        # Don't clear pending_title immediately - it might span lines
        if pending_title and not re.match(r'^(section|Amendment|Substitution)', line, re.IGNORECASE):
            pending_title = None
        
        i += 1
    
    # Save last clause
    if current_clause_num and current_clause_lines:
        clause_text = '\n'.join(current_clause_lines)
        if len(clause_text) > 100:
            clauses.append({
                'chapter': chapter_num,
                'chapter_title': chapter_title,
                'clause_num': current_clause_num,
                'clause_title': current_clause_title,
                'content': clause_text
            })
    
    return clauses

def chunk_clause(clause):
    """Convert a clause into one or more semantic chunks."""
    chunks = []
    
    clause_num = clause['clause_num']
    clause_title = clause['clause_title']
    chapter = clause['chapter'] or ''
    chapter_title = clause['chapter_title'] or ''
    content = clean_text(clause['content'])
    
    if len(content) < 50:
        return []
    
    # Build context prefix
    context = f"Clause {clause_num}"
    if clause_title:
        context += f": {clause_title}"
    context += "\n"
    if chapter:
        context += f"(Chapter {chapter}"
        if chapter_title:
            context += f" - {chapter_title}"
        context += ")\n"
    context += "\n"
    
    full_text = context + content
    section_amended = extract_section_being_amended(clause_title)
    
    if len(full_text) <= MAX_CHUNK_SIZE:
        chunks.append({
            'clause_num': clause_num,
            'clause_title': clause_title,
            'section_amended': section_amended,
            'chapter': chapter,
            'chapter_title': chapter_title,
            'text': full_text,
            'part': None
        })
    else:
        # Split by sub-clauses
        parts = re.split(r'(?=\n\s*\([a-z]\)|\n\s*\([ivxlc]+\))', content)
        
        current_chunk = []
        current_len = 0
        part_num = 1
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if current_len + len(part) > MAX_CHUNK_SIZE - len(context) and current_chunk:
                chunk_text = context + '\n'.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append({
                        'clause_num': clause_num,
                        'clause_title': clause_title,
                        'section_amended': section_amended,
                        'chapter': chapter,
                        'chapter_title': chapter_title,
                        'text': chunk_text,
                        'part': part_num
                    })
                    part_num += 1
                current_chunk = [part]
                current_len = len(part)
            else:
                current_chunk.append(part)
                current_len += len(part)
        
        if current_chunk:
            chunk_text = context + '\n'.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append({
                    'clause_num': clause_num,
                    'clause_title': clause_title,
                    'section_amended': section_amended,
                    'chapter': chapter,
                    'chapter_title': chapter_title,
                    'text': chunk_text,
                    'part': part_num if part_num > 1 else None
                })
    
    return chunks

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: {INPUT_PDF} not found")
        return

    print(f"Parsing clauses from {INPUT_PDF}...")
    clauses = parse_finance_bill(INPUT_PDF)
    print(f"Found {len(clauses)} clauses")
    
    if clauses:
        clause_nums = sorted(set(int(c['clause_num']) for c in clauses))
        print(f"Unique clause numbers: {len(clause_nums)}")
        print(f"Range: {min(clause_nums)} to {max(clause_nums)}")
    
    # Convert to chunks with deduplication
    all_chunks = []
    seen_doc_ids = {}
    
    for clause in clauses:
        chunks = chunk_clause(clause)
        for chunk in chunks:
            part_suffix = f"_part{chunk['part']}" if chunk['part'] else ""
            chapter_prefix = f"ch{chunk['chapter']}_" if chunk['chapter'] else ""
            
            if chunk['section_amended']:
                base_doc_id = f"fb2024_{chapter_prefix}clause{chunk['clause_num']}_sec{chunk['section_amended']}{part_suffix}"
            else:
                title_slug = slugify(chunk['clause_title']) if chunk['clause_title'] else ""
                base_doc_id = f"fb2024_{chapter_prefix}clause{chunk['clause_num']}"
                if title_slug:
                    base_doc_id += f"_{title_slug[:30]}"
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
                "doc_type": "finance_bill",
                "clause": chunk['clause_num'],
                "clause_title": chunk['clause_title'],
                "section_amended": chunk['section_amended'],
                "chapter": chunk['chapter'],
                "chapter_title": chunk['chapter_title'],
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
    print(f"Total clauses parsed: {len(clauses)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    if all_chunks:
        sizes = [len(c['text']) for c in all_chunks]
        print(f"Min chunk size: {min(sizes)} chars")
        print(f"Max chunk size: {max(sizes)} chars")
        print(f"Avg chunk size: {sum(sizes)//len(sizes)} chars")
        
        # Count by chapter
        chapters = {}
        for c in all_chunks:
            ch = c['chapter'] or 'Unknown'
            chapters[ch] = chapters.get(ch, 0) + 1
        print(f"\nChunks by chapter:")
        for ch, count in sorted(chapters.items()):
            print(f"  Chapter {ch}: {count}")
    
    # Sample output
    print("\n=== Sample Chunks ===")
    for chunk in all_chunks[:5]:
        print(f"  {chunk['doc_id']}: {chunk['text'][:80]}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
