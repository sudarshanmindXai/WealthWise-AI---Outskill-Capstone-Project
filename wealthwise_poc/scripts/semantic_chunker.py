"""
Semantic Chunker for Income Tax Act

Philosophy:
- Chunk by MEANING, not by structure
- One definition = one chunk (even if it spans many sub-clauses)
- Clean text: no headers, no "As amended by...", no page numbers
- doc_id reflects the concept, not just the clause number
"""

import json
import re
import os

# Configuration
INPUT_MD = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_act.md"
OUTPUT_JSONL = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_act_semantic.jsonl"

# Patterns
RE_DOCUMENT_HEADER = re.compile(r"^#\s+Income Tax Act", re.IGNORECASE)
RE_SOURCE_LINE = re.compile(r"^>\s*Source:", re.IGNORECASE)
RE_ACT_TITLE = re.compile(r"INCOME-T?AX ACT", re.IGNORECASE)
RE_AMENDED_BY = re.compile(r"\[?AS AMENDED BY.*?\]?", re.IGNORECASE)
RE_ACT_NUMBER = re.compile(r"\[\d+ OF \d+\]")
RE_CHAPTER = re.compile(r"^##\s*CHAPTER\s+([IXVLCDM]+[-A-Z]*)", re.IGNORECASE)
RE_SECTION = re.compile(r"^###\s*Section\s+(\d+[A-Z]*):?\s*(.*)", re.IGNORECASE)

# Definition start in Section 2: "(1)", "(1A)", "(2)", etc.
RE_DEFINITION_START = re.compile(r"^\((\d+[A-Z]*)\)\s*[\"']?([a-zA-Z][a-zA-Z\s\-]+)[\"']?\s*(means|includes|has the meaning|shall have)", re.IGNORECASE)
RE_CLAUSE_START = re.compile(r"^\((\d+[A-Z]*)\)\s+(.+)")

def slugify(text):
    """Create a clean slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')[:50]

def clean_metadata(text):
    """Remove document metadata, headers, act titles from text."""
    lines = text.split('\n')
    cleaned = []
    
    for line in lines:
        # Skip document headers
        if RE_DOCUMENT_HEADER.match(line):
            continue
        if RE_SOURCE_LINE.match(line):
            continue
        # Remove act title references
        line = RE_ACT_TITLE.sub('', line)
        line = RE_AMENDED_BY.sub('', line)
        line = RE_ACT_NUMBER.sub('', line)
        # Remove [***] markers
        line = re.sub(r'\[\*\*\*\]', '', line)
        # Remove footnote markers like "1[", "2[", etc. at start
        line = re.sub(r'^\d+\[', '[', line)
        # Clean up extra whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        
        if line:
            cleaned.append(line)
    
    return ' '.join(cleaned).strip()

def extract_definition_name(text):
    """Extract the term being defined from definition text."""
    # Look for patterns like: "agricultural income" means...
    match = re.search(r'^[\"\'"]?([a-zA-Z][a-zA-Z\s\-]+)[\"\'"]?\s*(means|includes|has the meaning)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"\'')
    return None

def parse_section_2_definitions(content_lines):
    """
    Special parser for Section 2 (Definitions).
    Each numbered clause (1), (1A), (2), etc. is ONE complete definition.
    All sub-clauses (a), (b), (i), etc. belong to the parent definition.
    """
    definitions = []
    current_def_num = None
    current_def_name = None
    current_def_text = []
    
    for line in content_lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a new top-level definition: (1), (1A), (2)...
        clause_match = RE_CLAUSE_START.match(line)
        if clause_match:
            clause_num = clause_match.group(1)
            # Is this a top-level definition (number or number+letter like 1A)?
            if re.match(r'^\d+[A-Z]?$', clause_num):
                # Save previous definition
                if current_def_num and current_def_text:
                    full_text = clean_metadata(' '.join(current_def_text))
                    if len(full_text) > 50:  # Skip tiny fragments
                        definitions.append({
                            'clause': current_def_num,
                            'name': current_def_name,
                            'text': full_text
                        })
                
                # Start new definition
                current_def_num = clause_num
                rest_of_line = clause_match.group(2)
                current_def_name = extract_definition_name(rest_of_line)
                current_def_text = [rest_of_line]
            else:
                # This is a sub-clause like (a), (i), (ii) - add to current definition
                current_def_text.append(line)
        else:
            # Continuation text
            if current_def_text is not None:
                current_def_text.append(line)
    
    # Don't forget the last definition
    if current_def_num and current_def_text:
        full_text = clean_metadata(' '.join(current_def_text))
        if len(full_text) > 50:
            definitions.append({
                'clause': current_def_num,
                'name': current_def_name,
                'text': full_text
            })
    
    return definitions

def parse_regular_section(section_num, section_title, content_lines):
    """
    Parser for non-definition sections.
    Groups content into meaningful legal chunks.
    """
    chunks = []
    full_text = clean_metadata(' '.join(content_lines))
    
    if len(full_text) < 50:
        return []
    
    # For now, keep entire section as one chunk if reasonable size
    # Split only if very large (>4000 chars)
    MAX_SIZE = 4000
    
    if len(full_text) <= MAX_SIZE:
        chunks.append({
            'section': section_num,
            'title': section_title,
            'text': full_text
        })
    else:
        # Split by sub-sections (1), (2), etc. but keep each sub-section complete
        # This is a fallback for very large sections
        parts = re.split(r'(?=\(\d+\)\s)', full_text)
        current_chunk = []
        current_len = 0
        
        for part in parts:
            if current_len + len(part) > MAX_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 50:
                    chunks.append({
                        'section': section_num,
                        'title': section_title,
                        'text': chunk_text,
                        'part': len(chunks) + 1
                    })
                current_chunk = [part]
                current_len = len(part)
            else:
                current_chunk.append(part)
                current_len += len(part)
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 50:
                chunks.append({
                    'section': section_num,
                    'title': section_title,
                    'text': chunk_text,
                    'part': len(chunks) + 1
                })
    
    return chunks

def main():
    if not os.path.exists(INPUT_MD):
        print(f"Error: {INPUT_MD} not found")
        return

    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_chunks = []
    
    current_chapter = None
    current_section_num = None
    current_section_title = None
    current_content = []
    in_preamble = True
    preamble_lines = []
    
    def flush_section():
        nonlocal current_section_num, current_section_title, current_content
        
        if not current_section_num or not current_content:
            return
        
        if current_section_num == "2":
            # Special handling for definitions
            definitions = parse_section_2_definitions(current_content)
            for defn in definitions:
                doc_id = f"it_act_2_{slugify(defn['name'] or defn['clause'])}"
                all_chunks.append({
                    "doc_id": doc_id,
                    "doc_type": "act",
                    "section": "2",
                    "sub_section": defn['name'] or f"clause_{defn['clause']}",
                    "chapter": current_chapter,
                    "ay": ["2025-26"],
                    "text": defn['text']
                })
        else:
            # Regular section
            chunks = parse_regular_section(current_section_num, current_section_title, current_content)
            for chunk in chunks:
                part_suffix = f"_part{chunk.get('part', '')}" if chunk.get('part') else ""
                doc_id = f"it_act_{current_section_num}{part_suffix}"
                all_chunks.append({
                    "doc_id": doc_id,
                    "doc_type": "act",
                    "section": current_section_num,
                    "sub_section": current_section_title,
                    "chapter": current_chapter,
                    "ay": ["2025-26"],
                    "text": chunk['text']
                })
        
        current_content = []

    for line in lines:
        stripped = line.strip()
        
        # Check for chapter
        chapter_match = RE_CHAPTER.match(stripped)
        if chapter_match:
            flush_section()
            current_chapter = chapter_match.group(1)
            in_preamble = False
            continue
        
        # Check for section
        section_match = RE_SECTION.match(stripped)
        if section_match:
            flush_section()
            current_section_num = section_match.group(1)
            current_section_title = section_match.group(2).strip() if section_match.group(2) else None
            in_preamble = False
            continue
        
        # Accumulate content
        if in_preamble:
            # Skip document metadata in preamble
            if RE_DOCUMENT_HEADER.match(stripped) or RE_SOURCE_LINE.match(stripped):
                continue
            if RE_ACT_TITLE.search(stripped) or RE_AMENDED_BY.search(stripped):
                continue
            if stripped:
                preamble_lines.append(stripped)
        else:
            if stripped and current_section_num:
                current_content.append(stripped)

    # Flush last section
    flush_section()
    
    # Add preamble if meaningful
    preamble_text = clean_metadata(' '.join(preamble_lines))
    if preamble_text and len(preamble_text) > 50:
        all_chunks.insert(0, {
            "doc_id": "it_act_preamble",
            "doc_type": "act",
            "section": "preamble",
            "sub_section": None,
            "chapter": None,
            "ay": ["2025-26"],
            "text": preamble_text
        })

    # Write output
    print(f"\nWriting {len(all_chunks)} semantic chunks to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    # Stats
    print("\n=== Semantic Chunking Stats ===")
    print(f"Total chunks: {len(all_chunks)}")
    
    section_2_chunks = [c for c in all_chunks if c['section'] == '2']
    print(f"Section 2 (Definitions): {len(section_2_chunks)} definitions")
    
    sizes = [len(c['text']) for c in all_chunks]
    print(f"Min size: {min(sizes)} chars")
    print(f"Max size: {max(sizes)} chars")
    print(f"Avg size: {sum(sizes)//len(sizes)} chars")
    
    # Sample definitions
    print("\n=== Sample Definitions from Section 2 ===")
    for chunk in section_2_chunks[:5]:
        print(f"  {chunk['sub_section']}: {chunk['text'][:80]}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
