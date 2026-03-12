import re
import tiktoken
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# Use the tokenizer that matches text-embedding-3-small
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_text(text: str, file_name: str = "") -> list[dict]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Returns list of dicts:
    [
      {
        "content":     "the actual chunk text",
        "chunk_index": 0,
        "token_count": 487,
        "metadata": {
          "file_name": "report.pdf",
          "section":   "Introduction"   # if heading was detected
        }
      },
      ...
    ]

    Strategy:
    1. Split into sentences (respects periods, newlines, headers)
    2. Accumulate sentences until we hit CHUNK_SIZE_TOKENS
    3. When limit hit: save chunk, backtrack CHUNK_OVERLAP_TOKENS
       so next chunk starts mid-previous-chunk (overlap)
    4. Track current heading/section for metadata context
    """
    if not text or not text.strip():
        return []

    sentences      = _split_into_sentences(text)
    chunks         = []
    current        = []
    current_tokens = 0
    current_section = ""
    chunk_index    = 0

    for sentence in sentences:
        # Detect section headings (lines starting with ## or all-caps short lines)
        if _is_heading(sentence):
            current_section = sentence.strip().lstrip("#").strip()

        sentence_tokens = count_tokens(sentence)

        # If single sentence exceeds chunk size, force it as its own chunk
        if sentence_tokens >= CHUNK_SIZE_TOKENS:
            if current:
                chunks.append(_make_chunk(current, chunk_index, file_name, current_section))
                chunk_index += 1
                current, current_tokens = [], 0

            chunks.append(_make_chunk([sentence], chunk_index, file_name, current_section))
            chunk_index += 1
            continue

        # If adding this sentence exceeds limit → save and start overlap
        if current_tokens + sentence_tokens > CHUNK_SIZE_TOKENS and current:
            chunks.append(_make_chunk(current, chunk_index, file_name, current_section))
            chunk_index += 1

            # Overlap: keep trailing sentences that fit within overlap budget
            current, current_tokens = _build_overlap(current)

        current.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current:
        chunks.append(_make_chunk(current, chunk_index, file_name, current_section))

    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a combination of:
    - Double newlines (paragraph breaks)
    - Single newlines for list items / headers
    - Period/question/exclamation followed by space + capital
    """
    # First split on paragraph breaks
    paragraphs = re.split(r"\n{2,}", text)
    sentences  = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split on sentence-ending punctuation followed by space + capital
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def _build_overlap(sentences: list[str]) -> tuple[list[str], int]:
    """
    From the END of the previous chunk, keep sentences that fit
    within CHUNK_OVERLAP_TOKENS. These become the start of the next chunk.
    """
    overlap        = []
    overlap_tokens = 0

    for sentence in reversed(sentences):
        t = count_tokens(sentence)
        if overlap_tokens + t > CHUNK_OVERLAP_TOKENS:
            break
        overlap.insert(0, sentence)
        overlap_tokens += t

    return overlap, overlap_tokens


def _make_chunk(sentences: list[str], index: int,
                file_name: str, section: str) -> dict:
    content = " ".join(sentences).strip()
    return {
        "content":     content,
        "chunk_index": index,
        "token_count": count_tokens(content),
        "metadata": {
            "file_name": file_name,
            "section":   section,
        }
    }


def _is_heading(text: str) -> bool:
    """Detect markdown headings or short all-caps lines."""
    text = text.strip()
    if text.startswith("#"):
        return True
    # Short line, all caps, no punctuation = likely a heading
    if len(text) < 80 and text.isupper() and not text.endswith("."):
        return True
    return False
