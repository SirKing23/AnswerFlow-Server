import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import hashlib
import os
import asyncio
from datetime import datetime, timezone

from utils.supabase_client import (
    download_file_from_storage,
    create_job,
    update_job_status,
    update_file_status,
    check_duplicate,
    store_embeddings,
)
from services.parser.router import parse_file
from chunker import chunk_text
from docling_chunker import chunk_markdown
from embedder import embed_chunks


async def process_file_pipeline(
    user_file_id: str,
    user_id:      str,
    storage_path: str,
    file_name:    str,
    mime_type:    str,
    bucket:       str,
):
    """
    Full pipeline: download → parse → chunk → embed → store.

    Status flow (written to both user_files and file_processing_jobs):
      queued → parsing → chunking → embedding → storing → done
      or any step → failed
    """
    job_id = None

    try:
        # ── 1. Create job record ──────────────────────────────────────
        await asyncio.sleep(5)
        job_id = await create_job(
            user_file_id=user_file_id,
            user_id=user_id,
            storage_path=storage_path,
            file_name=file_name,
            file_type=mime_type,
        )

        await update_file_status(user_file_id, "Queued", job_id=job_id)

        # ── 2. Download file from Supabase Storage ────────────────────
        await _set_status(user_file_id, job_id, "Parsing")
        file_bytes = await download_file_from_storage(storage_path, bucket)

        # ── 3. Deduplication check ────────────────────────────────────
        content_hash = hashlib.md5(file_bytes).hexdigest()
        is_duplicate = await check_duplicate(content_hash, user_id)

        if is_duplicate:
            await _set_status(user_file_id, job_id, "Skipped")
            
            await update_file_status(
                user_file_id, "Skipped",
                content_hash=content_hash,
                processed_at=True
            )
            return

        # ── 4. Parse ──────────────────────────────────────────────────
        text = await parse_file(file_bytes, file_name, mime_type)

        if not text or not text.strip():
            raise ValueError("Parsing produced empty text — file may be image-only or corrupted")

        # ── 5. Chunk ──────────────────────────────────────────────────
        await _set_status(user_file_id, job_id, "Chunking")
        # Try to use structured elements from parse_file if available
        # parse_file may return a JSON string (from docling) or plain markdown
        try:
            parsed = json.loads(text)
            if "elements" in parsed:
                # Rich structured output from docling_parser
                from docling_chunker import chunk_elements, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
                chunks = chunk_elements(
                    parsed["elements"],
                    source_filename=file_name,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    overlap=DEFAULT_OVERLAP,
                )
            else:
                raise ValueError("Not structured elements")
        except (json.JSONDecodeError, ValueError):
            # Fallback: plain markdown from old parser
            chunks = chunk_markdown(text, file_name=file_name)

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        if not chunks:
            raise ValueError("Chunking produced no chunks")


        print(f"[pipeline] Chunking complete: {len(chunks)} chunks created for file {file_name}")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk}")

        # # ── 6. Embed ──────────────────────────────────────────────────
        # await _set_status(user_file_id, job_id, "Embedding")
        # embedded_chunks = await embed_chunks(chunks)

        # # ── 7. Store embeddings ───────────────────────────────────────
        # await _set_status(user_file_id, job_id, "Storing")

        # rows = [
        #     {
        #         # Match your exact document_embeddings column names
        #         "user_id":     user_id,
        #         "job_id":      job_id,
        #         "file_path":   storage_path,
        #         "filename":    file_name,
        #         "file_type":   mime_type,
        #         "chunk_index": chunk["chunk_index"],
        #         "content":     chunk["content"],
        #         "embedding":   chunk["embedding"],
        #         "metadata":    chunk["metadata"],
        #         # inserted_at and created_at are handled by Supabase defaults
        #     }
        #     for chunk in embedded_chunks
        # ]

        # await store_embeddings(rows)

        # # ── 8. Mark done ──────────────────────────────────────────────
        # await update_job_status(job_id, "Completed", chunk_count=len(chunks))

        # await update_file_status(
        #     user_file_id, "Completed",
        #     content_hash=content_hash,
        #     processed_at=True
        # )

    except Exception as e:
        error_msg = str(e)
        print(f"[pipeline] ERROR for file {file_name}: {error_msg}")

        if job_id:
            await update_job_status(job_id, "Failed", error=error_msg)

        await update_file_status(user_file_id, "Failed")
        raise


async def _set_status(user_file_id: str, job_id: str, status: str):
    """Update status in both tables at once."""
    await update_job_status(job_id, status)
    await update_file_status(user_file_id, status)
