import io
import json
import httpx
from config import UNSTRUCTURED_API_KEY, UNSTRUCTURED_API_URL


async def parse_with_unstructured(file_bytes: bytes, file_name: str, mime_type: str) -> str:
    response = None
    last_error = None

    timeout = httpx.Timeout(
        connect=30.0,
        read=600.0,
        write=60.0,
        pool=30.0,
    )

    for strategy in ["hi_res", "fast"]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    UNSTRUCTURED_API_URL,  # Make sure this is the correct URL in config
                    headers={
                        "unstructured-api-key": UNSTRUCTURED_API_KEY,
                        "Accept": "application/json",
                    },
                    files={
                        "files": (file_name, io.BytesIO(file_bytes), mime_type)
                    },
                    data={
                        "strategy": strategy,
                        "include_page_breaks": "true",
                        "coordinates": "false",
                    },
                )

            if response.status_code == 200:
                break  # Success — stop retrying

            print(f"[unstructured] strategy={strategy} returned {response.status_code}: {response.text[:200]}")
            last_error = f"HTTP {response.status_code}: {response.text}"
            response = None  # Mark as failed so the post-loop check catches it

        except (httpx.ReadError, httpx.TimeoutException) as e:
            print(f"[unstructured] strategy={strategy} timed out or failed: {e}")
            last_error = str(e)
            response = None

    if response is None or response.status_code != 200:
        raise ValueError(
            f"Unstructured.io failed on all strategies. Last error: {last_error}"
        )

    raw_elements = response.json()

    if not raw_elements:
        raise ValueError("Unstructured.io returned no elements — file may be empty or unreadable")

    return _elements_to_structured_json(raw_elements, file_name)


def _elements_to_structured_json(raw_elements: list[dict], file_name: str) -> str:
    """
    Convert Unstructured API elements into the structured JSON format
    expected by docling_chunker.chunk_elements(), preserving page numbers.

    Returns a JSON string with {"filename", "elements"} matching the
    docling_parser / PyMuPDF output format.
    """
    skip_types = {"Footer", "Header", "PageBreak"}
    heading_types = {"Title"}

    elements = []
    seen_pages = set()

    for el in raw_elements:
        el_type = el.get("type", "")
        text = el.get("text", "").strip()
        metadata = el.get("metadata", {})
        page = metadata.get("page_number")

        if page is not None:
            seen_pages.add(page)

        if not text or el_type in skip_types:
            continue

        if el_type in heading_types:
            elements.append({
                "type":  "heading",
                "level": 2,
                "text":  text,
                "page":  page,
            })

        elif el_type == "Table":
            elements.append({
                "type":     "table",
                "markdown": text,
                "text":     text,
                "page":     page,
            })

        elif el_type == "ListItem":
            elements.append({
                "type": "text",
                "text": f"• {text}",
                "page": page,
            })

        elif el_type == "Image":
            elements.append({
                "type": "picture",
                "annotations": [text] if text else [],
                "page": page,
            })

        else:
            # NarrativeText, UncategorizedText, etc.
            elements.append({
                "type": "text",
                "text": text,
                "page": page,
            })

    num_pages = max(seen_pages) if seen_pages else None

    return json.dumps({
        "filename":  file_name,
        "num_pages": num_pages,
        "elements":  elements,
    }, ensure_ascii=False)