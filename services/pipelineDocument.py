"""
pipelineDocument.py  (updated — adds Neo4j knowledge graph step)
=================================================================
Full pipeline: download → parse → chunk → embed → store → graph

New step 8 (between store embeddings and mark done):
  For each embedded chunk → extract entities + relations → store in Neo4j Aura

The graph step is non-blocking — a failure there marks the file "CompletedPartial"
so the user still gets vector search even if graph extraction hits an error.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import hashlib
import asyncio
import logging
from datetime import datetime, timezone

from utils.supabase_client import (
    download_file_from_storage,
    create_job,
    update_job_status,
    update_file_status,
    check_duplicate,
    store_embeddings,
)
from utils.neo4j_client import store_graph_data, delete_graph_for_file
from services.parser.router import parse_file
from services.entity_extractor import extract_entities_and_relations
from chunker import chunk_text
from docling_chunker import chunk_markdown
from embedder import embed_chunks
from config import GRAPH_CONCURRENCY

log = logging.getLogger(__name__)


async def process_file_pipeline(
    user_file_id: str,
    user_id:      str,
    storage_path: str,
    file_name:    str,
    mime_type:    str,
    bucket:       str,
):
    """
    Full pipeline: download → parse → chunk → embed → store → graph.

    Status flow:
      queued → parsing → chunking → embedding → storing → graphing → done
      or any step → failed
      graph failure only → CompletedPartial  (vector search still works)
    """
    job_id = None

    try:
        # ── 1. Create job record ──────────────────────────────────────────────
        await asyncio.sleep(5)   # race-condition guard (see RAG_SYSTEM_DESIGN.md)
        job_id = await create_job(
            user_file_id=user_file_id,
            user_id=user_id,
            storage_path=storage_path,
            file_name=file_name,
            file_type=mime_type,
        )
        await update_file_status(user_file_id, "Queued", job_id=job_id)

        # ── 2. Download file from Supabase Storage ────────────────────────────
        await _set_status(user_file_id, job_id, "Parsing")
        file_bytes = await download_file_from_storage(storage_path, bucket)

        # ── 3. Deduplication check ────────────────────────────────────────────
        content_hash = hashlib.md5(file_bytes).hexdigest()
        is_duplicate = await check_duplicate(content_hash, user_id)

        if is_duplicate:
            await _set_status(user_file_id, job_id, "Skipped")
            await update_file_status(
                user_file_id, "Skipped",
                content_hash=content_hash,
                processed_at=True,
            )
            return

        # ── 4. Parse ──────────────────────────────────────────────────────────
        text = await parse_file(file_bytes, file_name, mime_type)

        if not text or not text.strip():
            raise ValueError("Parsing produced empty text — file may be image-only or corrupted")

        # ── 5. Chunk ──────────────────────────────────────────────────────────
        await _set_status(user_file_id, job_id, "Chunking")

        try:
            parsed = json.loads(text)
            if "elements" in parsed:
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
            chunks = chunk_markdown(text, file_name=file_name)

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        # ── 6. Embed ──────────────────────────────────────────────────────────
        await _set_status(user_file_id, job_id, "Embedding")
        embedded_chunks = await embed_chunks(chunks)

        # ── 7. Store embeddings → Supabase pgvector ───────────────────────────
        await _set_status(user_file_id, job_id, "Storing")

        rows = [
            {
                "user_id":     user_id,
                "job_id":      job_id,
                "file_path":   storage_path,
                "filename":    file_name,
                "file_type":   mime_type,
                "chunk_index": chunk["chunk_index"],
                "content":     chunk["content"],
                "embedding":   chunk["embedding"],
                "metadata":    chunk["metadata"],
            }
            for chunk in embedded_chunks
        ]
        
        await store_embeddings(rows)

        # ── 8. Build knowledge graph → Neo4j Aura ────────────────────────────
        # Runs after vector storage so a graph failure never blocks retrieval.
        await _set_status(user_file_id, job_id, "Graphing")
        graph_ok = await _build_knowledge_graph(
            chunks=embedded_chunks,
            user_id=user_id,
            file_name=file_name,
            job_id=job_id,
        )

        # ── 9. Mark done ──────────────────────────────────────────────────────
        final_status = "Completed" if graph_ok else "CompletedPartial"
        await update_job_status(job_id, final_status, chunk_count=len(chunks))
        await update_file_status(
            user_file_id, final_status,
            content_hash=content_hash,
            processed_at=True,
        )
        log.info(
            f"[pipeline] Done: {file_name} | {len(chunks)} chunks | "
            f"graph={'ok' if graph_ok else 'partial'}"
        )

    except Exception as e:
        error_msg = str(e)
        log.error(f"[pipeline] ERROR for file {file_name}: {error_msg}")

        if job_id:
            await update_job_status(job_id, "Failed", error=error_msg)

        await update_file_status(user_file_id, "Failed")
        raise


# ── Graph builder helper ──────────────────────────────────────────────────────

async def _build_knowledge_graph(
    chunks:    list[dict],
    user_id:   str,
    file_name: str,
    job_id:    str,
) -> bool:
    """
    Extract entities + relations from every chunk and store them in Neo4j.

    Processes GRAPH_CONCURRENCY chunks in parallel to balance speed vs rate limits.
    Returns True if ALL chunks succeeded, False if any chunk partially failed
    (the pipeline continues in both cases — vector search is unaffected).
    """
    semaphore = asyncio.Semaphore(GRAPH_CONCURRENCY)
    errors    = 0

    async def _process_chunk(chunk: dict) -> None:
        nonlocal errors
        async with semaphore:
            try:
                graph_data = await extract_entities_and_relations(
                                chunk_text=chunk["content"],
                                file_name=file_name,
                                metadata=chunk.get("metadata", {}),
                            )

                if graph_data["entities"] or graph_data["relations"]:
                    await store_graph_data(
                        user_id=user_id,
                        file_name=file_name,
                        job_id=job_id,
                        chunk_index=chunk["chunk_index"],
                        entities=graph_data["entities"],
                        relations=graph_data["relations"],
                    )
                    log.debug(
                        f"[pipeline] graph chunk {chunk['chunk_index']}: "
                        f"{len(graph_data['entities'])} entities, "
                        f"{len(graph_data['relations'])} relations"
                    )
            except Exception as e:
                errors += 1
                log.warning(f"[pipeline] graph error on chunk {chunk['chunk_index']}: {e}")
                # If the connection dropped, reset the singleton so the next
                # chunk gets a fresh driver instead of reusing a dead one.
                if "defunct connection" in str(e).lower() or isinstance(e, ConnectionResetError):
                    try:
                        from utils.neo4j_client import close_neo4j
                        await close_neo4j()
                        log.info("[pipeline] Neo4j driver reset after defunct connection")
                    except Exception:
                        pass

    await asyncio.gather(*[_process_chunk(c) for c in chunks])

    if errors:
        log.warning(f"[pipeline] Graph extraction: {errors}/{len(chunks)} chunks had errors")

    return errors == 0


async def _set_status(user_file_id: str, job_id: str, status: str):
    """Update status in both tables simultaneously."""
    await update_job_status(job_id, status)
    await update_file_status(user_file_id, status)