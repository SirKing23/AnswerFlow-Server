import io
import httpx
from config import UNSTRUCTURED_API_KEY, UNSTRUCTURED_API_URL


async def parse_with_unstructured(file_bytes: bytes, file_name: str, mime_type: str) -> str:
    """
    Send file to Unstructured.io API and return clean extracted text.

    Unstructured returns a list of elements like:
    [
      {"type": "Title",     "text": "Introduction"},
      {"type": "NarrativeText", "text": "This paper discusses..."},
      {"type": "Table",     "text": "Col1 | Col2 ..."},
      ...
    ]

    We preserve element type context by prepending it to the text,
    which significantly improves retrieval accuracy for structured docs.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            UNSTRUCTURED_API_URL,
            headers={
                "unstructured-api-key": UNSTRUCTURED_API_KEY,
                "Accept": "application/json",
            },
            files={
                "files": (file_name, io.BytesIO(file_bytes), mime_type)
            },
            data={
                # hi_res gives better table/layout extraction
                # fast is cheaper but misses complex layouts
                "strategy":             "hi_res",
                "include_page_breaks":  "true",
                "coordinates":          "false",   # we don't need bounding boxes
            }
        )

    if response.status_code != 200:
        raise ValueError(
            f"Unstructured.io API error {response.status_code}: {response.text}"
        )

    elements = response.json()

    if not elements:
        raise ValueError("Unstructured.io returned no elements — file may be empty or unreadable")

    return _elements_to_text(elements)


def _elements_to_text(elements: list[dict]) -> str:
    """
    Convert Unstructured elements into clean readable text.

    Element types we handle:
    - Title / Header      → prepend as section context
    - NarrativeText       → main content, include as-is
    - ListItem            → include with bullet prefix
    - Table               → include with [TABLE] marker
    - PageBreak           → insert double newline
    - Footer / Header     → skip (usually noise)
    - UncategorizedText   → include if not empty
    """
    skip_types = {"Footer", "Header", "PageBreak"}
    section_types = {"Title", "Header"}

    lines = []
    current_section = None

    for el in elements:
        el_type = el.get("type", "")
        text    = el.get("text", "").strip()

        if not text or el_type in skip_types:
            continue

        if el_type in section_types:
            # Track current section so chunks can reference it
            current_section = text
            lines.append(f"\n## {text}")

        elif el_type == "Table":
            lines.append(f"\n[TABLE]\n{text}\n[/TABLE]")

        elif el_type == "ListItem":
            lines.append(f"• {text}")

        elif el_type == "PageBreak":
            lines.append("\n")

        else:
            # NarrativeText, UncategorizedText, etc.
            lines.append(text)

    return "\n".join(lines).strip()
