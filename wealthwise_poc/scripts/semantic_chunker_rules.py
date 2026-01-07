"""
Semantic Chunker for Income Tax Rules 1962

Philosophy (same as Act chunker):
- Chunk by MEANING, not by structure
- One complete rule = one chunk (unless very large)
- Clean text: no headers, no metadata noise
- doc_id reflects the rule number and topic
"""

import json
import re
import os

# Configuration
INPUT_MD = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/Income_Tax_Rules_Final.md"
OUTPUT_JSONL = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_rules_semantic.jsonl"

# Max chunk size before splitting (Rules can be very long, e.g., Rule 3 - Valuation of perquisites)
MAX_CHUNK_SIZE = 6000  # Larger than Act because rules are more self-contained

# Patterns
RE_RULE_HEADER = re.compile(r"^##\s*Rule\s*[-–]\s*(\d+[A-Z]*)", re.IGNORECASE)
RE_SUB_RULE_START = re.compile(r"^\((\d+[A-Z]*)\)\s+")
RE_DOCUMENT_HEADER = re.compile(r"^#\s+Income Tax Rules", re.IGNORECASE)
RE_NOTIFICATION = re.compile(r"\[NOTIFICATION.*?\]", re.IGNORECASE)
RE_HORIZONTAL_RULE = re.compile(r"^-{3,}$")

def slugify(text):
    """Create a clean slug from text."""
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')[:50]

def clean_text(text):
    """Remove metadata and noise from text."""
    # Remove notification references
    text = RE_NOTIFICATION.sub('', text)
    # Remove footnote markers like "1[", "2[" etc.
    text = re.sub(r'^\d+\[', '[', text)
    text = re.sub(r'\s\d+\[', ' [', text)
    # Remove [***] markers
    text = re.sub(r'\[\*\*\*\]', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def extract_rule_title(first_lines):
    """Extract the title/subject of the rule from its first few lines."""
    for line in first_lines[:5]:
        line = line.strip()
        if line and not RE_RULE_HEADER.match(line) and not line.startswith('---'):
            # Skip lines that are just sub-rule starts
            if RE_SUB_RULE_START.match(line):
                continue
            # This is likely the title
            if len(line) < 200:  # Reasonable title length
                return line.split('.')[0].strip()  # Take first sentence
    return None

def parse_rules(lines):
    """Parse the markdown file into rules."""
    rules = []
    current_rule = None
    current_rule_num = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip document header
        if RE_DOCUMENT_HEADER.match(stripped):
            continue
        
        # Skip horizontal rules
        if RE_HORIZONTAL_RULE.match(stripped):
            continue
        
        # Check for new rule
        rule_match = RE_RULE_HEADER.match(stripped)
        if rule_match:
            # Save previous rule
            if current_rule_num and current_content:
                title = extract_rule_title(current_content)
                rules.append({
                    'rule_num': current_rule_num,
                    'title': title,
                    'content': current_content
                })
            
            # Start new rule
            current_rule_num = rule_match.group(1)
            current_content = []
            continue
        
        # Accumulate content
        if current_rule_num is not None:
            current_content.append(line.rstrip())
    
    # Don't forget the last rule
    if current_rule_num and current_content:
        title = extract_rule_title(current_content)
        rules.append({
            'rule_num': current_rule_num,
            'title': title,
            'content': current_content
        })
    
    return rules

def chunk_rule(rule):
    """Convert a rule into one or more semantic chunks."""
    chunks = []
    rule_num = rule['rule_num']
    title = rule['title']
    full_text = '\n'.join(rule['content'])
    full_text = clean_text(full_text)
    
    if len(full_text) < 100:
        return []  # Skip very short rules
    
    # Create context prefix
    context_prefix = f"Rule {rule_num}"
    if title:
        context_prefix += f": {title}"
    context_prefix += "\n\n"
    
    if len(full_text) <= MAX_CHUNK_SIZE:
        # Single chunk
        chunks.append({
            'rule': rule_num,
            'title': title,
            'text': context_prefix + full_text,
            'part': None
        })
    else:
        # Need to split - split by sub-rules (1), (2), etc.
        # First, try to find natural split points
        parts = re.split(r'(?=^\(\d+[A-Z]*\)\s)', full_text, flags=re.MULTILINE)
        
        current_chunk = []
        current_len = 0
        part_num = 1
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if current_len + len(part) > MAX_CHUNK_SIZE and current_chunk:
                # Flush current chunk
                chunk_text = context_prefix + '\n'.join(current_chunk)
                if len(chunk_text) > 100:
                    chunks.append({
                        'rule': rule_num,
                        'title': title,
                        'text': chunk_text,
                        'part': part_num
                    })
                    part_num += 1
                current_chunk = [part]
                current_len = len(part)
            else:
                current_chunk.append(part)
                current_len += len(part)
        
        # Flush remaining
        if current_chunk:
            chunk_text = context_prefix + '\n'.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append({
                    'rule': rule_num,
                    'title': title,
                    'text': chunk_text,
                    'part': part_num if part_num > 1 else None
                })
    
    return chunks

def main():
    if not os.path.exists(INPUT_MD):
        print(f"Error: {INPUT_MD} not found")
        return

    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Parsing rules...")
    rules = parse_rules(lines)
    print(f"Found {len(rules)} rules")
    
    # Convert to chunks
    all_chunks = []
    for rule in rules:
        chunks = chunk_rule(rule)
        for chunk in chunks:
            part_suffix = f"_part{chunk['part']}" if chunk['part'] else ""
            title_slug = slugify(chunk['title']) if chunk['title'] else ""
            
            doc_id = f"it_rules_{chunk['rule']}"
            if title_slug:
                doc_id += f"_{title_slug}"
            doc_id += part_suffix
            
            all_chunks.append({
                "doc_id": doc_id,
                "doc_type": "rules",
                "rule": chunk['rule'],
                "title": chunk['title'],
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
    print(f"Total rules parsed: {len(rules)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    sizes = [len(c['text']) for c in all_chunks]
    print(f"Min chunk size: {min(sizes)} chars")
    print(f"Max chunk size: {max(sizes)} chars")
    print(f"Avg chunk size: {sum(sizes)//len(sizes)} chars")
    
    # Sample output
    print("\n=== Sample Chunks ===")
    for chunk in all_chunks[:5]:
        print(f"  {chunk['doc_id']}: {chunk['text'][:80]}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
