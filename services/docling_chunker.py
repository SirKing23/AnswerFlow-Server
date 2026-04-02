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

Token settings (tuned for OpenAI text-embedding-3-small):
  CHUNK_SIZE_TOKENS    = 512   (max tokens per chunk)
  CHUNK_OVERLAP_TOKENS = 100   (~20% overlap to preserve cross-boundary context)

Usage:
  python docling_chunker.py report_meta.json
  python docling_chunker.py report.md
  python docling_chunker.py report_meta.json --chunk-size 256
  python docling_chunker.py report_meta.json --overlap 50
  python docling_chunker.py report_meta.json --output-dir ./chunks
  python docling_chunker.py *.json --output-dir ./chunks
"""

import argparse
import itertools
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

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
# Spinner
# ---------------------------------------------------------------------------
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = ""):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._current_line_len = 0

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            line = f"\r  {frame}  {self.message}"
            sys.stdout.write(line)
            sys.stdout.flush()
            self._current_line_len = len(line)
            time.sleep(0.08)

    def start(self):
        self._thread.start()
        return self

    def stop(self, final_msg: str = "", status: str = "OK"):
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write(f"\r{' ' * (self._current_line_len + 2)}\r")
        sys.stdout.flush()
        if final_msg:
            icon = "✔" if status == "OK" else "✖" if status == "ERROR" else "⚠"
            print(f"  {icon}  {final_msg}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk docling-parsed documents into embedder-ready JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="_meta.json or .md file(s) produced by docling_parser.py",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=f"Max tokens per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        metavar="N",
        help=f"Overlap tokens between chunks (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Output directory (default: same folder as input file).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load content from _meta.json or .md
# ---------------------------------------------------------------------------
def load_elements(input_path: Path) -> tuple[list[dict], str]:
    """
    Returns (elements, source_filename).

    For _meta.json: reads the 'elements' array directly — each element has
    type, text, page, headings etc. (produced by _build_content_json).

    For .md: parses Markdown into synthetic elements so the same
    chunking logic applies.
    """
    ext = input_path.suffix.lower()

    if ext == ".json":
        raw = json.loads(input_path.read_text(encoding="utf-8"))

        # Validate it's our content JSON format (has 'elements' key)
        if "elements" not in raw:
            raise RuntimeError(
                "This JSON doesn't look like a docling_parser.py content JSON "
                "(missing 'elements' key). Make sure you're passing the "
                "_meta.json produced by the updated docling_parser.py."
            )

        source = raw.get("filename", input_path.stem.replace("_meta", ""))
        return raw["elements"], source

    elif ext in (".md", ".markdown"):
        elements = _parse_markdown_to_elements(
            input_path.read_text(encoding="utf-8")
        )
        return elements, input_path.name

    else:
        raise RuntimeError(
            f"Unsupported file type '{ext}'. Pass a _meta.json or .md file."
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
# Public API — chunk a markdown string produced by parse_with_docling
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


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------
def process_file(input_path: Path, args) -> bool:
    print(f"  {'─'*56}")
    print(f"  📄  {input_path.name}")
    print(f"  {'─'*56}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    stem = input_path.stem
    if stem.endswith("_meta"):
        stem = stem[:-5]
    output_path = output_dir / f"{stem}_chunks.json"

    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    elements = None
    source   = None
    err      = None

    def do_load():
        nonlocal elements, source, err
        try:
            elements, source = load_elements(input_path)
        except Exception as exc:
            err = exc

    with Spinner("Loading content JSON…"):
        t = threading.Thread(target=do_load, daemon=True)
        t.start(); t.join()

    if err:
        print(f"  ✖  Failed to load: {err}")
        return False

    print(f"  ✔  Loaded {len(elements)} elements from {source}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks  = None
    err     = None

    def do_chunk():
        nonlocal chunks, err
        try:
            chunks = chunk_elements(
                elements, source, args.chunk_size, args.overlap
            )
        except Exception as exc:
            err = exc

    with Spinner(
        f"Chunking… (size={args.chunk_size} tokens, overlap={args.overlap} tokens)"
    ):
        t = threading.Thread(target=do_chunk, daemon=True)
        t.start(); t.join()

    if err:
        print(f"  ✖  Chunking failed: {err}")
        return False

    elapsed = time.time() - t0
    print(f"  ✔  {len(chunks)} chunks created in {elapsed:.1f}s")

    # ── Token stats ───────────────────────────────────────────────────────────
    if chunks:
        token_counts = [c["token_count"] for c in chunks]
        avg = sum(token_counts) / len(token_counts)
        print(f"  ℹ  Tokens — avg: {avg:.0f}  min: {min(token_counts)}  max: {max(token_counts)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with Spinner("Saving chunks.json…"):
        output_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"  ✔  Saved → {output_path.name}")

    # ── Sample preview ────────────────────────────────────────────────────────
    if chunks:
        s = chunks[0]
        print(f"\n  📦  Sample — chunk #0:")
        print(f"      section      : {s['metadata']['section'] or '(none)'}")
        print(f"      page         : {s['metadata']['page'] or '?'}")
        print(f"      token_count  : {s['token_count']}")
        print(f"      element_types: {s['metadata']['element_types']}")
        preview = s['content'][:150] + ("…" if len(s['content']) > 150 else "")
        print(f"      content      : {preview}")

    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve inputs
    input_paths = []
    for raw in args.inputs:
        p = Path(raw)
        if "*" in str(p) or "?" in str(p):
            input_paths.extend(sorted(Path(".").glob(str(p))))
        elif p.exists():
            input_paths.append(p)
        else:
            print(f"  ⚠  File not found: {raw}")

    if not input_paths:
        print("  ✖  No valid input files found.")
        sys.exit(1)

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║           DOCLING DOCUMENT CHUNKER                      ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Files       : {len(input_paths)}")
    print(f"  Chunk size  : {args.chunk_size} tokens")
    print(f"  Overlap     : {args.overlap} tokens")
    print(f"  Embedding   : OpenAI text-embedding-3-small compatible ✔")
    print()

    success, failed = 0, 0
    for i, path in enumerate(input_paths, 1):
        if len(input_paths) > 1:
            print(f"  [{i} of {len(input_paths)}]")
        ok = process_file(path, args)
        if ok:
            success += 1
        else:
            failed += 1

    print("  ╔══════════════════════════════════════════════════════════╗")
    summary = f"  ✔ {success} succeeded    ✖ {failed} failed"
    print(f"  ║  All done!  {summary:<47}║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Next step — pass chunks to your embedder.py:")
    print()
    print("    import json, asyncio")
    print("    from embedder import embed_chunks")
    print()
    stem_example = input_paths[0].stem.replace("_meta", "")
    print(f"    chunks   = json.load(open('{stem_example}_chunks.json'))")
    print("    embedded = asyncio.run(embed_chunks(chunks))")
    print()


if __name__ == "__main__":
    main()