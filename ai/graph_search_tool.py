"""
graph_search_tool.py
=====================
Drop this file ANYWHERE in your project (e.g. services/graph_search_tool.py)
then import the two exports into tools.py:

    from services.graph_search_tool import (
        graph_search_tool,
        entity_explorer_tool,
    )

And register them in orchestrator.py:

    tools=[
        query_decomposer_tool,
        user_query_embedder_tool,
        search_chunks_tool,
        graph_search_tool,        # ← new
        entity_explorer_tool,     # ← new
        context_builder_tool,
        answer_validator_tool,
    ]

These two tools give the agent relationship-aware retrieval on top of the
existing vector search. The agent decides when to call them based on the query.
"""

import sys
import os
import json
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import function_tool
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
from utils.neo4j_client import graph_search, get_entity_neighbors

log = logging.getLogger(__name__)

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Shared run context reference ──────────────────────────────────────────────
# tools.py owns _run_context — import it rather than duplicating it.
# This import works because tools.py is always loaded first by orchestrator.py.
try:
    from tools import _run_context
except ImportError:
    # Fallback for unit tests / direct imports
    _run_context: dict = {}


# ── Helper: extract entity names from free text ───────────────────────────────

async def _extract_query_entities(text: str) -> list[str]:
    """
    Quick NER pass on the user's raw question to pull entity names.
    We only need names (not types) for graph lookup — so this is cheaper
    than a full extract_entities_and_relations() call.
    """
    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract named entities (people, companies, places, products, "
                    "concepts) from the user's question. "
                    "Return ONLY a JSON array of strings — entity names only, nothing else. "
                    "Example: [\"Acme Corp\", \"Manila\", \"Q3 2024\"]\n"
                    "If there are no clear named entities, return []."
                )
            },
            {"role": "user", "content": text[:500]},
        ],
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        # Model may return {"entities": [...]} or just [...]
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v if x]
    except Exception:
        pass
    return []


# ── Tool 1: graph_search_tool ─────────────────────────────────────────────────

class GraphSearchInput(BaseModel):
    run_id:     str = Field(description="The current run ID for context sharing")
    user_query: str = Field(description="The user's question or a sub-query to search the graph for")
    max_hops:   int = Field(default=2, description="How many relationship hops to traverse (1 or 2)")


@function_tool
async def graph_search_tool(input: GraphSearchInput) -> str:
    """
    Searches the Neo4j knowledge graph for documents and entities related to the query.

    Use this AFTER search_chunks_tool when:
    - The question mentions specific people, companies, or named concepts
    - The question asks about relationships ("how does X relate to Y?")
    - Vector search returned few or irrelevant results
    - The question is entity-centric ("all docs mentioning Acme Corp")

    Returns file names and matched/related entities so the agent can
    cross-reference with vector search results for a richer answer.
    """
    ctx     = _run_context.get(input.run_id, {})
    user_id = ctx.get("user_id")

    if not user_id:
        return "Error: No user_id in context."

    # Extract entity names from the query
    entity_names = await _extract_query_entities(input.user_query)

    if not entity_names:
        return (
            "No named entities found in the query — graph search requires at least one "
            "named entity (person, company, concept, etc.). "
            "Try search_chunks_tool for semantic search instead."
        )

    log.debug(f"[graph_search_tool] query entities: {entity_names}")

    results = await graph_search(
        user_id=user_id,
        entity_names=entity_names,
        max_hops=input.max_hops,
        limit=8,
    )

    if not results:
        return (
            f"No documents found in the knowledge graph for entities: "
            f"{', '.join(entity_names)}. "
            "These entities may not have been extracted during ingestion, "
            "or the documents haven't been processed yet."
        )

    # Store graph results in run context for context_builder_tool to use
    ctx["graph_results"] = results
    ctx["graph_entities"] = entity_names
    _run_context[input.run_id] = ctx

    lines = [f"Knowledge graph results for: {', '.join(entity_names)}\n"]
    for r in results:
        matched  = ", ".join(r.get("matched_entities", []))
        related  = ", ".join(r.get("related_entities", [])[:5])  # cap display
        score    = r.get("relevance_score", 0)
        lines.append(
            f"File: {r['file_name']} | Score: {score}\n"
            f"  Matched entities: {matched}\n"
            f"  Related entities: {related or 'none'}\n"
        )

    return "\n".join(lines)


# ── Tool 2: entity_explorer_tool ──────────────────────────────────────────────

class EntityExploreInput(BaseModel):
    run_id:      str = Field(description="The current run ID for context sharing")
    entity_name: str = Field(description="The exact entity name to explore (as it appears in documents)")


@function_tool
async def entity_explorer_tool(input: EntityExploreInput) -> str:
    """
    Returns all entities directly connected to a named entity in the knowledge graph.

    Use this when:
    - The user asks "what is X connected to?" or "tell me about X"
    - You found an entity in graph_search_tool and want to explore its neighborhood
    - You want to discover related topics before running vector search

    Returns neighbor entities with their relationship types.
    """
    ctx     = _run_context.get(input.run_id, {})
    user_id = ctx.get("user_id")

    if not user_id:
        return "Error: No user_id in context."

    neighbors = await get_entity_neighbors(
        user_id=user_id,
        entity_name=input.entity_name,
        limit=15,
    )

    if not neighbors:
        return (
            f"No connections found for entity '{input.entity_name}'. "
            "The entity may not exist in the knowledge graph yet, "
            "or it may have been stored under a slightly different name."
        )

    lines = [f"Connections for '{input.entity_name}':\n"]
    for n in neighbors:
        lines.append(
            f"  {n['neighbor']} ({n['type']})  — via {n['relation']}"
        )

    return "\n".join(lines)
