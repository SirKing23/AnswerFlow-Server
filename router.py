import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import SELF_PARSE_MIME_TYPES, UNSTRUCTURED_MIME_TYPES
from services.parser.text_parser   import parse_txt, parse_markdown, parse_csv
from services.parser.docx_parser   import parse_docx
from services.parser.excel_parser  import parse_xlsx
from services.parser.unstructured_parser import parse_with_unstructured


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
    if mime_type in UNSTRUCTURED_MIME_TYPES:
        return await parse_with_unstructured(file_bytes, file_name, mime_type)

    # ── Unknown mime type ─────────────────────────────────────────────
    raise ValueError(f"Unsupported mime type: {mime_type}")
