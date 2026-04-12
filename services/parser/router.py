import os
import sys

import fitz  # PyMuPDF
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import SELF_PARSE_MIME_TYPES, UNSTRUCTURED_MIME_TYPES
from text_parser import parse_txt, parse_markdown, parse_csv
from docx_parser import parse_docx
from excel_parser import parse_xlsx
from unstructured_parser import parse_with_unstructured
from docling_parser import parse_with_docling




async def parse_file(file_bytes: bytes, file_name: str, mime_type: str) -> str:
    """
    Route file to the correct parser based on mime type.

    Self-parse (free, fast):
      - txt, markdown, csv  → text_parser
      - docx                → mammoth
      - xlsx, xls           → openpyxl

    Unstructured.io (uses page quota):
      - pdf                 → unstructured (hi_res strategy)
      - doc (old binary)    → unstructured
      - pptx                → unstructured
    """

    # ── Self-parse path ───────────────────────────────────────────────
    if mime_type in SELF_PARSE_MIME_TYPES:

        if mime_type in ("text/plain",):
            return parse_txt(file_bytes)

        if mime_type == "text/markdown":
            return parse_markdown(file_bytes)

        if mime_type in ("text/csv", "text/tsv", "text/tab-separated-values"):
            return parse_csv(file_bytes)

        if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return parse_docx(file_bytes)

        if mime_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ):
            return parse_xlsx(file_bytes)

    # ── Unstructured.io path ──────────────────────────────────────────
    #  if mime_type in UNSTRUCTURED_MIME_TYPES:
    #     return await parse_with_unstructured(file_bytes, file_name, mime_type)
    
    # ── docling path ──────────────────────────────────────────
    #if mime_type in UNSTRUCTURED_MIME_TYPES:
    #    return await parse_with_docling(file_bytes, file_name, mime_type)
    
    if mime_type in UNSTRUCTURED_MIME_TYPES:
       return await parse_pdf_smart(file_bytes, file_name)



async def parse_pdf_smart(file_bytes: bytes, file_name: str) -> str:
    # Try PyMuPDF first — now returns structured JSON with page numbers
    text = _parse_pdf_pymupdf(file_bytes, file_name)

    # Check if elements were actually extracted
    import json
    parsed = json.loads(text)
    total_text = " ".join(e.get("text", "") for e in parsed["elements"])

    if len(total_text.strip()) > 200:
        print(f"[router] PyMuPDF succeeded: {len(parsed['elements'])} elements")
        return text  # structured JSON → pipelineDocument.py picks this up correctly

    # Scanned PDF fallback
    print(f"[router] Scanned PDF detected — falling back to Unstructured.io")
    return await parse_with_unstructured(file_bytes, file_name, "application/pdf")


def _parse_pdf_pymupdf(file_bytes: bytes, file_name: str) -> str:
    """
    Parse PDF and return structured JSON matching docling_parser output format
    so docling_chunker.py gets page numbers on every element.
    """
    import fitz
    import json

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    elements = []

    for page_num, page in enumerate(doc, start=1):  # start=1 matches docling page numbering
        blocks = page.get_text("blocks")  # returns list of (x0,y0,x1,y1,text,block_no,block_type)

        for block in blocks:
            text = block[4].strip()
            block_type = block[6]  # 0=text, 1=image

            if not text or block_type != 0:
                continue

            # Detect if this block looks like a heading
            lines = text.split("\n")
            first_line = lines[0].strip()
            is_heading = (
                len(first_line) < 80
                and len(lines) == 1
                and not first_line.endswith(".")
                and (first_line.isupper() or first_line.istitle())
            )

            if is_heading:
                elements.append({
                    "type":  "heading",
                    "level": 2,
                    "text":  first_line,
                    "page":  page_num,   # ← page number attached here
                })
                # remaining lines after heading as text
                remainder = "\n".join(lines[1:]).strip()
                if remainder:
                    elements.append({
                        "type": "text",
                        "text": remainder,
                        "page": page_num,
                    })
            else:
                elements.append({
                    "type": "text",
                    "text": text,
                    "page": page_num,   # ← page number attached here
                })

        # Extract tables via pdfplumber for this page
        tables = _extract_tables_from_page(file_bytes, page_num - 1)  # pdfplumber is 0-indexed
        for table_text in tables:
            elements.append({
                "type": "text",
                "text": table_text,
                "page": page_num,       # ← page number attached here
            })

    doc.close()

    # Return same JSON structure docling_parser returns
    # so pipelineDocument.py and docling_chunker.py work without any changes
    return json.dumps({
        "filename": file_name,
        "num_pages": len(doc) if not doc.is_closed else None,
        "elements":  elements,
    }, ensure_ascii=False)

def _extract_tables_from_page(file_bytes: bytes, page_num: int) -> list[str]:
    """Use pdfplumber for table extraction on a specific page."""
    import io
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if page_num >= len(pdf.pages):
                return []
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            result = []
            for table in tables:
                if not table or not table[0]:
                    continue
                headers = [str(c or "").strip() for c in table[0]]
                for row in table[1:]:
                    if all(cell is None for cell in row):
                        continue
                    readable = " | ".join(
                        f"{headers[i]}: {str(cell).strip()}"
                        for i, cell in enumerate(row)
                        if i < len(headers) and cell
                    )
                    if readable:
                        result.append(readable)
            return result
    except Exception:
        return []

