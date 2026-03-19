import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import function_tool
from pydantic import BaseModel, Field
from typing import Optional
from services.embedder import embed_single
from utils.supabase_client import get_supabase
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Shared run context ────────────────────────────────────────────────────────
# Module-level dict keyed by run_id so tools share state within one agent run.
# e.g. embedding created in user_query_embedder_tool is reused by search_chunks_tool
_run_context: dict = {}


def set_run_context(run_id: str, user_id: str, file_id: Optional[str]):
    _run_context[run_id] = {
        "user_id":   user_id,
        "file_id":   file_id,
        "embedding": None,
        "chunks":    [],
        "enriched_context": "",
        "sources":   [],
        "decomposed_queries": "",
    }


def get_run_context(run_id: str) -> dict:
    return _run_context.get(run_id, {})


def clear_run_context(run_id: str):
    _run_context.pop(run_id, None)


# ── Tool 1: user_query_embedder_tool ─────────────────────────────────────────

class EmbedInput(BaseModel):
    run_id: str = Field(description="The current run ID for context sharing")
    text:   str = Field(description="The text to embed")


@function_tool
async def user_query_embedder_tool(input: EmbedInput) -> str:
    """
    Embeds any text into a vector using OpenAI text-embedding-3-small.
    Always call this first before search_chunks_tool.
    Stores the embedding in run context so search_chunks_tool can reuse it.
    Returns a confirmation that embedding was created.
    """
    embedding = await embed_single(input.text)
    ctx = _run_context.get(input.run_id, {})
    ctx["embedding"] = embedding
    _run_context[input.run_id] = ctx
    return f"Embedding created successfully ({len(embedding)} dimensions). Ready for search."


# ── Tool 2: search_chunks_tool ────────────────────────────────────────────────

class SearchInput(BaseModel):
    run_id:               str   = Field(description="The current run ID for context sharing")
    similarity_threshold: float = Field(default=0.70, description="Minimum similarity score 0-1")
    match_count:          int   = Field(default=5,    description="Number of chunks to retrieve")


@function_tool
async def search_chunks_tool(input: SearchInput) -> str:
    """
    Searches the user's vector database for chunks relevant to the embedded query.
    Must call user_query_embedder_tool first to create the embedding.
    Respects file_id scoping — if file_id was provided, only searches that file.
    Returns the top matching chunks with their source file and similarity score.
    """
    ctx       = _run_context.get(input.run_id, {})
    embedding = ctx.get("embedding")
    user_id   = ctx.get("user_id")
    file_id   = ctx.get("file_id")

    if not embedding:
        return "Error: No embedding found. Call user_query_embedder_tool first."
    if not user_id:
        return "Error: No user_id in context."

    supabase = get_supabase()

    if file_id:
        result = supabase.rpc("match_chunks_by_file", {
            "query_embedding":      embedding,
            "match_user_id":        user_id,
            "match_file_id":        file_id,
            "match_count":          input.match_count,
            "similarity_threshold": input.similarity_threshold,
        }).execute()
    else:
        result = supabase.rpc("match_chunks", {
            "query_embedding":      embedding,
            "match_user_id":        user_id,
            "match_count":          input.match_count,
            "similarity_threshold": input.similarity_threshold,
        }).execute()

    chunks = result.data or []

    if not chunks:
        return "No relevant chunks found in the user's documents for this query."

    ctx["chunks"] = chunks
    _run_context[input.run_id] = ctx

    lines = [f"Found {len(chunks)} relevant chunks:\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Chunk {i}] File: {chunk['filename']} | "
            f"Similarity: {round(chunk['similarity'], 3)}\n"
            f"{chunk['content'][:300]}{'...' if len(chunk['content']) > 300 else ''}\n"
        )
    return "\n".join(lines)


# ── Tool 3: query_decomposer_tool ─────────────────────────────────────────────
# Breaks complex multi-part questions into focused sub-queries.
# Useful when user asks compound questions like:
# "Compare section 3 and section 5 and tell me which is more important"

class DecomposeInput(BaseModel):
    run_id: str = Field(description="The current run ID for context sharing")
    query:  str = Field(description="The user's original question to decompose")


@function_tool
async def query_decomposer_tool(input: DecomposeInput) -> str:
    """
    Analyzes the user's query and breaks it into focused sub-questions if complex.
    Use this when the query is multi-part, ambiguous, or requires multiple searches.
    Returns either the original query (if simple) or a numbered list of sub-queries.
    """
    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query analysis expert. Analyze the given question and determine "
                    "if it needs to be broken into sub-questions for better document retrieval. "
                    "If the question is simple and focused, return it as-is. "
                    "If it is complex or multi-part, break it into 2-4 focused sub-questions. "
                    "Return ONLY the question(s), numbered if multiple, nothing else."
                )
            },
            {"role": "user", "content": input.query}
        ],
        temperature=0.1,
        max_tokens=300,
    )
    result = response.choices[0].message.content.strip()
    ctx = _run_context.get(input.run_id, {})
    ctx["decomposed_queries"] = result
    _run_context[input.run_id] = ctx
    return result


# ── Tool 4: context_builder_tool ──────────────────────────────────────────────

class ContextBuilderInput(BaseModel):
    run_id:       str  = Field(description="The current run ID for context sharing")
    user_query:   str  = Field(description="The user's original question")
    chat_history: list = Field(default=[], description="Previous conversation messages")


@function_tool
async def context_builder_tool(input: ContextBuilderInput) -> str:
    """
    Analyzes retrieved chunks alongside user query and chat history to build
    a rich coherent context. Filters irrelevant chunks, resolves pronoun
    references from history, and structures context for optimal LLM comprehension.
    Always call this after search_chunks_tool and before generating the final answer.
    """
    ctx    = _run_context.get(input.run_id, {})
    chunks = ctx.get("chunks", [])

    if not chunks:
        return "No chunks available. Run search_chunks_tool first."

    raw_context = "\n\n---\n\n".join([
        f"[Source: {c['filename']} | Chunk {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    ])

    history_text = ""
    if input.chat_history:
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in input.chat_history[-6:]
        ])

    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a context analyst. Given retrieved document chunks, "
                    "a user query, and optional chat history, your job is to:\n"
                    "1. Filter out chunks clearly irrelevant to the query\n"
                    "2. Resolve pronouns or references using chat history\n"
                    "3. Organize remaining chunks logically\n"
                    "4. Return cleaned structured context ready for answering\n"
                    "Keep all source labels intact. Do not add information not in the chunks."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Query: {input.user_query}\n\n"
                    f"Chat History:\n{history_text or 'None'}\n\n"
                    f"Retrieved Chunks:\n{raw_context}"
                )
            }
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    enriched_context = response.choices[0].message.content.strip()

    ctx["enriched_context"] = enriched_context
    ctx["sources"] = [
        {
            "filename":    c["filename"],
            "chunk_index": c["chunk_index"],
            "similarity":  round(c["similarity"], 3),
            "excerpt":     c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"]
        }
        for c in chunks
    ]
    _run_context[input.run_id] = ctx
    return enriched_context


# ── Tool 5: answer_validator_tool ─────────────────────────────────────────────
# Validates the final answer is grounded in retrieved context.
# Catches hallucinations — claims not supported by source documents.
# Returns VALID or lists specific unsupported claims for the orchestrator to fix.

class ValidatorInput(BaseModel):
    run_id: str = Field(description="The current run ID for context sharing")
    answer: str = Field(description="The proposed answer to validate")


@function_tool
async def answer_validator_tool(input: ValidatorInput) -> str:
    """
    Validates that the proposed answer is fully grounded in retrieved document chunks.
    Detects hallucinations — claims not supported by the source documents.
    Returns 'VALID' if grounded, or lists unsupported claims to fix.
    Always call this before returning the final answer to the user.
    """
    ctx     = _run_context.get(input.run_id, {})
    context = ctx.get("enriched_context", "")

    if not context:
        return "VALID (no context to validate against)"

    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. Given context from documents "
                    "and a proposed answer, check if every claim in the answer is "
                    "supported by the context. "
                    "Reply with exactly 'VALID' if fully grounded. "
                    "Otherwise list each unsupported claim starting with '- UNSUPPORTED:'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nAnswer to validate:\n{input.answer}"
            }
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
