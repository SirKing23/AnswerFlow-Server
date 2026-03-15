import io
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

    elements = response.json()

    if not elements:
        raise ValueError("Unstructured.io returned no elements — file may be empty or unreadable")

    return _elements_to_text(elements)


def _elements_to_text(elements: list[dict]) -> str:
    skip_types = {"Footer", "Header", "PageBreak"}
    section_types = {"Title", "Header"}

    lines = []

    for el in elements:
        el_type = el.get("type", "")
        text = el.get("text", "").strip()

        if not text or el_type in skip_types:
            continue

        if el_type in section_types:
            lines.append(f"\n## {text}")
        elif el_type == "Table":
            lines.append(f"\n[TABLE]\n{text}\n[/TABLE]")
        elif el_type == "ListItem":
            lines.append(f"• {text}")
        elif el_type == "PageBreak":
            lines.append("\n")
        else:
            lines.append(text)

    return "\n".join(lines).strip()