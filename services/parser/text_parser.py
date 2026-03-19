import csv
import io


def parse_txt(file_bytes: bytes) -> str:
    """Plain text — decode and return as-is."""
    return file_bytes.decode("utf-8", errors="replace").strip()


def parse_markdown(file_bytes: bytes) -> str:
    """
    Markdown — strip common syntax markers so embeddings focus on content,
    not symbols like ##, **, __, etc.
    """
    import re
    text = file_bytes.decode("utf-8", errors="replace")

    # Remove code blocks first (preserve content inside)
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)

    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Remove links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove images
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


def parse_csv(file_bytes: bytes) -> str:
    """
    CSV — convert each row into a readable sentence.
    Headers become keys, values become values.

    Example row: {"Name": "John", "Age": "30"} 
    becomes: "Name: John | Age: 30"

    This embeds much better than raw CSV text.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        # Skip completely empty rows
        if not any(row.values()):
            continue
        # Build "Key: Value | Key: Value" format
        readable = " | ".join(
            f"{k.strip()}: {v.strip()}"
            for k, v in row.items()
            if k and v and v.strip()
        )
        if readable:
            rows.append(readable)

    return "\n".join(rows)
