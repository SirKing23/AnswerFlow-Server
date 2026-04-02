#!/usr/bin/env python3
"""
Docling Document Chunker
=========================
Chunks documents already parsed by docling_parser.py into embedder-ready JSON.

Accepts:
  - filename_meta.json  — produced by docling_parser.py (recommended, has full text)
  - filename.md         — also produced by docling_parser.py (fallback)

Output:
  - filename_chunks.json — fully compatible with embedder.py (embed_chunks)

Chunk format (identical to your existing chunker.py pipeline):
  {
    "content":     "the actual chunk text",
    "chunk_index": 0,
    "token_count": 487,
    "metadata": {
      "file_name": "report.pdf",
      "section":   "Introduction",
      "headings":  ["Chapter 1", "Introduction"],
      "page":      3,
      "element_types": ["text", "text", "table"]
    }
  }



"""


import os
import re
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# ---------------------------------------------------------------------------
# Default token settings — tuned for text-embedding-3-small
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE    = CHUNK_SIZE_TOKENS    # tokens per chunk
DEFAULT_OVERLAP       = CHUNK_OVERLAP_TOKENS  # overlap tokens between consecutive chunks


# ---------------------------------------------------------------------------
# Token counter — matches chunker.py (1 token ≈ 4 chars)
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(text) // 4


# ---------------------------------------------------------------------------
# Core chunker — works directly on the elements list
# ---------------------------------------------------------------------------
def chunk_elements(
    elements: list[dict],
    source_filename: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Walk through document elements and produce overlapping chunks.

    Strategy:
    - Tracks current heading/section context as elements are visited
    - Accumulates element text until chunk_size is reached
    - On overflow: saves chunk, backtracks 'overlap' tokens for next chunk
    - Tables are kept whole (never split mid-table)
    - Picture annotations are included as descriptive text
    - Each chunk records which page(s) it covers and its heading context
    """

    # ── Step 1: flatten elements into "units" (text + metadata) ──────────────
    units = []           # list of {text, page, element_type, is_heading}
    current_headings = []

    for el in elements:
        etype = el.get("type", "text")

        if etype == "heading":
            level = el.get("level", 1)
            text  = el.get("text", "").strip()
            if not text:
                continue
            # Maintain heading stack by level
            current_headings = [
                h for h in current_headings if h["level"] < level
            ]
            current_headings.append({"level": level, "text": text})
            units.append({
                "text":         text,
                "page":         el.get("page"),
                "element_type": "heading",
                "is_heading":   True,
                "headings":     [h["text"] for h in current_headings],
            })

        elif etype == "text":
            text = el.get("text", "").strip()
            if not text:
                continue
            units.append({
                "text":         text,
                "page":         el.get("page"),
                "element_type": "text",
                "is_heading":   False,
                "headings":     [h["text"] for h in current_headings],
            })

        elif etype == "table":
            # Use markdown representation; fall back to stringified data
            text = el.get("markdown", "").strip()
            if not text and el.get("data"):
                rows = el["data"]
                text = "\n".join(" | ".join(str(c) for c in row) for row in rows)
            if not text:
                continue
            units.append({
                "text":         text,
                "page":         el.get("page"),
                "element_type": "table",
                "is_heading":   False,
                "headings":     [h["text"] for h in current_headings],
            })

        elif etype == "picture":
            annotations = el.get("annotations", [])
            if not annotations:
                continue
            text = " ".join(annotations).strip()
            if not text:
                continue
            units.append({
                "text":         f"[Figure: {text}]",
                "page":         el.get("page"),
                "element_type": "picture",
                "is_heading":   False,
                "headings":     [h["text"] for h in current_headings],
            })

    if not units:
        return []

    # ── Step 2: accumulate units into chunks with overlap ─────────────────────
    chunks        = []
    chunk_index   = 0
    i             = 0   # current position in units list

    while i < len(units):
        current_units  = []
        current_tokens = 0
        current_pages  = set()

        # Fill chunk up to chunk_size
        j = i
        while j < len(units):
            unit   = units[j]
            tokens = count_tokens(unit["text"])

            # A single unit larger than chunk_size → force it as its own chunk
            if tokens >= chunk_size and not current_units:
                current_units.append(unit)
                if unit["page"] is not None:
                    current_pages.add(unit["page"])
                j += 1
                break

            # Would overflow → stop filling
            if current_tokens + tokens > chunk_size and current_units:
                break

            current_units.append(unit)
            current_tokens += tokens
            if unit["page"] is not None:
                current_pages.add(unit["page"])
            j += 1

        if not current_units:
            i += 1
            continue

        # Build chunk content
        content = "\n\n".join(u["text"] for u in current_units).strip()

       # Use headings from the chunk's heading context — never from tables/text
        # Find the most recent heading seen across all units in this chunk
        headings = []
        for u in current_units:
            if u["element_type"] == "heading":
                headings = u["headings"]
            elif u["headings"] and not headings:
                headings = u["headings"]
        # Clean headings — remove any that look like table content
        headings = [
            h for h in headings
            if len(h) < 200 and not h.strip().startswith("|")
        ]
        section = headings[-1] if headings else ""

        # Collect element types for metadata transparency
        element_types = [u["element_type"] for u in current_units]

        # Page range
        page = min(current_pages) if current_pages else None

        chunks.append({
            "content":     content,
            "chunk_index": chunk_index,
            "token_count": count_tokens(content),
            "metadata": {
                "file_name":     source_filename,
                "section":       section,
                "headings":      headings,
                "page":          page,
                "element_types": element_types,
            }
        })
        chunk_index += 1

        # ── Overlap: backtrack so next chunk starts overlap tokens before j ──
        overlap_tokens = 0
        next_start = j - 1
        while next_start > i:
            t = count_tokens(units[next_start]["text"])
            if overlap_tokens + t > overlap:
                break
            overlap_tokens += t
            next_start -= 1

        # Move forward by at least 1 to guarantee progress
        i = max(i + 1, next_start + 1)

    return chunks

# ---------------------------------------------------------------------------
# chunk a markdown string produced by parse_with_docling
# ---------------------------------------------------------------------------
def chunk_markdown(markdown_text: str, file_name: str = "") -> list[dict]:
    """
    Convert a markdown string (output of parse_with_docling) into
    embedder-ready chunks using the same structured chunking logic.

    Returns the same list[dict] format as chunker.chunk_text.
    """
    elements = _parse_markdown_to_elements(markdown_text)
    return chunk_elements(
        elements,
        source_filename=file_name,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )

def _parse_markdown_to_elements(md_text: str) -> list[dict]:
    """Convert a Markdown string into a list of synthetic elements."""
    elements = []
    lines = md_text.splitlines()
    buffer = []

    def flush_buffer():
        text = " ".join(buffer).strip()
        if text:
            elements.append({"type": "text", "text": text, "page": None})
        buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buffer()
            continue

        # Markdown heading
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            flush_buffer()
            elements.append({
                "type":  "heading",
                "level": len(m.group(1)),
                "text":  m.group(2).strip(),
                "page":  None,
            })
            continue

        # Markdown table row
        if stripped.startswith("|"):
            flush_buffer()
            elements.append({
                "type":     "table",
                "markdown": stripped,
                "page":     None,
            })
            continue

        buffer.append(stripped)

    flush_buffer()
    return elements


