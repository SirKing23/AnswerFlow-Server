import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks using OpenAI text-embedding-3-small.

    Batches up to EMBEDDING_BATCH_SIZE chunks per API call
    (OpenAI allows up to 2048 inputs per request).

    Returns the same chunks list with 'embedding' key added to each.
    """
    if not chunks:
        return []

    # Split into batches
    batches = [
        chunks[i: i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE)
    ]

    results = []
    for batch in batches:
        texts      = [chunk["content"] for chunk in batch]
        embeddings = await _embed_batch(texts)

        for chunk, embedding in zip(batch, embeddings):
            results.append({**chunk, "embedding": embedding})

    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI embeddings API for a batch of texts.
    Retries up to 3 times with exponential backoff on failure.
    """
    response = await _client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    # Response order matches input order — guaranteed by OpenAI
    return [item.embedding for item in response.data]


async def embed_single(text: str) -> list[float]:
    """Embed a single string — used for RAG query embedding."""
    embeddings = await _embed_batch([text])
    return embeddings[0]
