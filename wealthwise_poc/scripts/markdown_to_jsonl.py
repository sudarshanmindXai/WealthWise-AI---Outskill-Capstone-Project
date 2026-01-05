import json
import re
import os
from collections import defaultdict

# Configuration
INPUT_MD = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_act.md"
OUTPUT_JSONL = "/Users/sid_viscious/Desktop/Folders/WealthWise-AI---Outskill-Capstone-Project/wealthwise_poc/income_tax_docs/tier_1/income_tax_act_optimized.jsonl"

# Thresholds
MAX_CHUNK_SIZE = 2000  # chars (approx 500 words)
MIN_CHUNK_SIZE = 100   # Filter out tiny chunks

# Patterns for noise
RE_OMITTED = re.compile(r"^\s*\[\*\*\*\]\s*$")  # Standalone [***] lines
RE_MOSTLY_OMITTED = re.compile(r"^\s*\(\d+[A-Za-z]*\)\s*\[\*\*\*\]")  # e.g., "(3) [***]"

def slugify(text):
    return re.sub(r'[\W_]+', '_', text).lower().strip('_')

def is_noise_chunk(text):
    """Check if a chunk is mostly noise (omitted sections, footnotes)."""
    stripped = text.strip()
    # If it's just [***] or short with only [***]
    if RE_OMITTED.match(stripped):
        return True
    # If the majority of the chunk is [***] markers
    if stripped.count('[***]') > 3 and len(stripped) < 200:
        return True
    return False

def clean_text(text):
    """Remove standalone [***] lines but keep context around them."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Skip lines that are just [***]
        if RE_OMITTED.match(line):
            continue
        # Skip lines like "(3) [***]" - omitted subsections
        if RE_MOSTLY_OMITTED.match(line.strip()):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def create_doc_id(chapter, section, part=0):
    s_chap = slugify(chapter) if chapter else "preamble"
    s_sec = slugify(section.split(":")[0]) if section else "intro"
    base = f"it_act_{s_chap}_{s_sec}"
    if part > 0:
        return f"{base}_part{part}"
    return base

def main():
    if not os.path.exists(INPUT_MD):
        print("Markdown file not found.")
        return

    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_chunks = []
    
    current_chapter = None
    current_section_title = None
    current_content_buffer = []

    def flush_buffer():
        nonlocal current_content_buffer
        if not current_content_buffer:
            return
            
        full_text = "\n".join(current_content_buffer).strip()
        # Clean [***] noise
        full_text = clean_text(full_text)
        
        if not full_text or len(full_text) < MIN_CHUNK_SIZE:
            current_content_buffer = []
            return
        
        # Skip if it's a noise chunk
        if is_noise_chunk(full_text):
            current_content_buffer = []
            return

        # Context prefix
        context_prefix = ""
        if current_chapter:
            context_prefix += f"Chapter {current_chapter}. "
        if current_section_title:
            context_prefix += f"{current_section_title}\n\n"
        
        if len(full_text) > MAX_CHUNK_SIZE:
            # Split logic
            parts = []
            current_part = []
            current_len = 0
            
            paragraphs = full_text.split('\n') 
            
            for p in paragraphs:
                if current_len + len(p) > MAX_CHUNK_SIZE and current_part:
                    parts.append("\n".join(current_part))
                    current_part = []
                    current_len = 0
                current_part.append(p)
                current_len += len(p)
            
            if current_part:
                parts.append("\n".join(current_part))
                
            for i, part in enumerate(parts):
                final_text = (context_prefix + part).strip()
                if len(final_text) < MIN_CHUNK_SIZE:
                    continue
                    
                raw_chunks.append({
                    "chapter": current_chapter,
                    "section": current_section_title,
                    "text": final_text,
                    "part": i + 1
                })
        else:
            final_text = (context_prefix + full_text).strip()
            raw_chunks.append({
                "chapter": current_chapter,
                "section": current_section_title,
                "text": final_text,
                "part": 0
            })
            
        current_content_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("## CHAPTER"):
            flush_buffer()
            current_chapter = line.replace("## CHAPTER", "").strip()
            current_section_title = None
            continue
            
        if line.startswith("### Section"):
            flush_buffer()
            current_section_title = line.replace("### ", "").strip()
            continue
            
        current_content_buffer.append(line)

    # Final flush
    flush_buffer()
    
    # De-duplicate doc_ids
    doc_id_counts = defaultdict(int)
    final_chunks = []
    
    for chunk in raw_chunks:
        base_id = create_doc_id(chunk['chapter'], chunk['section'], chunk['part'])
        
        # Check for collision
        if doc_id_counts[base_id] > 0:
            # Append a unique suffix
            doc_id = f"{base_id}_{doc_id_counts[base_id]}"
        else:
            doc_id = base_id
        
        doc_id_counts[base_id] += 1
        
        final_chunks.append({
            "doc_id": doc_id,
            "doc_type": "act",
            "section": chunk['section'],
            "chapter": chunk['chapter'],
            "text": chunk['text'],
            "ay": ["2025-26"]
        })

    # Write JSONL
    print(f"Writing {len(final_chunks)} chunks to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk) + "\n")

    # Stats
    print(f"\n=== Quality Stats ===")
    print(f"Total chunks: {len(final_chunks)}")
    sizes = [len(c['text']) for c in final_chunks]
    print(f"Min size: {min(sizes)} chars")
    print(f"Max size: {max(sizes)} chars")
    print(f"Avg size: {sum(sizes)//len(sizes)} chars")
    
    # Verify uniqueness
    ids = [c['doc_id'] for c in final_chunks]
    print(f"Unique doc_ids: {len(set(ids))}")
    print(f"Duplicates: {len(ids) - len(set(ids))}")
    
    # Check for [***]
    omitted = sum(1 for c in final_chunks if '[***]' in c['text'])
    print(f"Chunks with [***]: {omitted}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
