import io
import re
import mammoth


def parse_docx(file_bytes: bytes) -> str:
    """
    Parse a .docx file using mammoth.
    
    mammoth.extract_raw_text() gives us clean plain text while 
    respecting the document structure (paragraphs, lists, headings).
    
    We use raw text over HTML conversion because:
    - We don't need HTML tags for embeddings
    - Raw text is cleaner for chunking
    - Headings are preserved as plain text lines
    """
    result = mammoth.extract_raw_text(io.BytesIO(file_bytes))

    text = result.value

    # Collapse more than 2 consecutive newlines into 2
    # (mammoth sometimes produces excessive whitespace between sections)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
