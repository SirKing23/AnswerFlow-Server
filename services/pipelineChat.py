import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from utils.supabase_client import get_supabase
from services.embedder import embed_single
from config import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
    CHAT_TEMPERATURE, CHAT_MAX_TOKENS,
    SEARCH_TOP_K, SEARCH_SIMILARITY_THRESHOLD,
)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# How many chunks to retrieve for context
TOP_K = SEARCH_TOP_K
SIMILARITY_THRESHOLD = SEARCH_SIMILARITY_THRESHOLD


async def chat_with_docs(
    user_id: str,
    message: str,
    history: list,
    file_id: str | None,
) -> dict:
    """
    Full RAG chat pipeline:
    1. Embed the user's question
    2. Search document_embeddings via pgvector (scoped to file or all files)
    3. Build context from top-K chunks
    4. Send to GPT-4o-mini with conversation history
    5. Return answer + sources
    """

    # ── 1. Embed the user's question ──────────────────────────────────
    query_embedding = await embed_single(message)

    # ── 2. Vector similarity search ───────────────────────────────────
    chunks = await search_chunks(
        user_id=user_id,
        query_embedding=query_embedding,
        file_id=file_id,
    )

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in your documents to answer that question.",
            "sources": []
        }

    # ── 3. Build context from retrieved chunks ────────────────────────
    context = build_context(chunks)

    # ── 4. Build messages array with history ──────────────────────────
    messages = build_messages(message, history, context)

    # ── 5. Call GPT-4o-mini ───────────────────────────────────────────
    response = await _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS,
    )

    answer = response.choices[0].message.content

    # ── 6. Format sources for frontend ───────────────────────────────
    sources = [
        {
            "filename":    chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "similarity":  round(chunk["similarity"], 3),
            "excerpt":     chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
        }
        for chunk in chunks
    ]

    return {
        "answer":  answer,
        "sources": sources,
    }


async def search_chunks(
    user_id: str,
    query_embedding: list[float],
    file_id: str | None,
) -> list[dict]:
    """
    Search document_embeddings using pgvector cosine similarity.

    If file_id is provided: scope to chunks from that specific file only.
    If file_id is None: search across ALL of the user's files.
    """
    supabase = get_supabase()

    if file_id:
        # Scope to specific file — join through file_processing_jobs
        result = supabase.rpc("match_chunks_by_file", {
            "query_embedding":    query_embedding,
            "match_user_id":      user_id,
            "match_file_id":      file_id,
            "match_count":        TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }).execute()
    else:
        # Search all user's files
        result = supabase.rpc("match_chunks", {
            "query_embedding":    query_embedding,
            "match_user_id":      user_id,
            "match_count":        TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }).execute()

    return result.data or []


def build_context(chunks: list[dict]) -> str:
    """
    Assemble retrieved chunks into a readable context block.
    Each chunk is labeled with its source file so the LLM can cite it.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['filename']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(
    message: str,
    history: list,
    context: str,
) -> list[dict]:
    """
    Build the full messages array for OpenAI:
    - System prompt with RAG instructions
    - Conversation history (previous turns)
    - Current user message with context injected
    """
    system_prompt = """You are a helpful assistant that answers questions based on the user's uploaded documents.

Rules:
- Answer ONLY using the provided context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly.
- Always cite which source (Source 1, Source 2, etc.) your answer comes from.
- Be concise and direct. Use bullet points for lists.
- If asked a follow-up question, use the conversation history to maintain context."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (previous turns)
    for msg in history:
        messages.append({
            "role":    msg.role,
            "content": msg.content,
        })

    # Add current question with context injected
    messages.append({
        "role": "user",
        "content": f"""Context from documents:

{context}

---

Question: {message}"""
    })

    return messages
