import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import function_tool
from pydantic import BaseModel, Field
from typing import Optional
from services.embedder import embed_single
from utils.supabase_client import get_supabase
from openai import AsyncOpenAI
from config import (
    OPENAI_API_KEY,
    DECOMPOSER_TEMPERATURE, DECOMPOSER_MAX_TOKENS,
    CONTEXT_BUILDER_TEMPERATURE, CONTEXT_BUILDER_MAX_TOKENS,
    VALIDATOR_TEMPERATURE, VALIDATOR_MAX_TOKENS,
    QUERY_NER_TEMPERATURE, QUERY_NER_MAX_TOKENS,
    TEXT2CYPHER_TEMPERATURE, TEXT2CYPHER_MAX_TOKENS,
    SEARCH_TOP_K, SEARCH_SIMILARITY_THRESHOLD,
    GRAPH_SEARCH_LIMIT, ENTITY_EXPLORER_LIMIT,
    TEXT2CYPHER_DEFAULT_LIMIT, TEXT2CYPHER_MAX_LIMIT,
    CHAT_HISTORY_WINDOW,
)

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Shared run context ────────────────────────────────────────────────────────
_run_context: dict = {}


def set_run_context(run_id: str, user_id: str, file_id: Optional[str]):
    _run_context[run_id] = {
        "user_id":            user_id,
        "file_id":            file_id,
        "embedding":          None,
        "chunks":             [],
        "enriched_context":   "",
        "sources":            [],
        "decomposed_queries": "",
        "graph_results":      [],
        "graph_entities":     [],
    }


def get_run_context(run_id: str) -> dict:
    return _run_context.get(run_id, {})


def clear_run_context(run_id: str):
    _run_context.pop(run_id, None)


# ── Logging helpers ───────────────────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)

def _ok(msg: str):   print(f"  OK   {msg}")
def _warn(msg: str): print(f"  WARN {msg}")
def _info(msg: str): print(f"       {msg}")
def _end():          print("=" * 62 + "\n")


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
    """
    embedding = await embed_single(input.text)
    ctx = _run_context.get(input.run_id, {})
    ctx["embedding"] = embedding
    _run_context[input.run_id] = ctx
    print(f"[embedder] {len(embedding)}d vector created for: {input.text[:60]}")
    return f"Embedding created ({len(embedding)} dimensions). Ready for search."


# ── Tool 2: search_chunks_tool ────────────────────────────────────────────────

class SearchInput(BaseModel):
    run_id:               str   = Field(description="The current run ID for context sharing")
    similarity_threshold: float = Field(default=SEARCH_SIMILARITY_THRESHOLD, description="Minimum similarity score 0-1")
    match_count:          int   = Field(default=SEARCH_TOP_K,    description="Number of chunks to retrieve")


@function_tool
async def search_chunks_tool(input: SearchInput) -> str:
    """
    Searches the user's pgvector database for chunks semantically similar to the embedded query.
    Must call user_query_embedder_tool first.
    Respects file_id scoping — searches one file or all files depending on context.
    """
    ctx       = _run_context.get(input.run_id, {})
    embedding = ctx.get("embedding")
    user_id   = ctx.get("user_id")
    file_id   = ctx.get("file_id")

    if not embedding:
        return "Error: No embedding found. Call user_query_embedder_tool first."
    if not user_id:
        return "Error: No user_id in context."

    _banner("SUPABASE pgvector HIT -- search_chunks_tool")
    _info(f"user_id   : {user_id}")
    _info(f"file_id   : {file_id or 'all files'}")
    _info(f"threshold : {input.similarity_threshold}  |  top-k: {input.match_count}")

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
        _warn("pgvector returned 0 chunks")
        _end()
        return "No relevant chunks found in the user's documents for this query."

    _ok(f"pgvector returned {len(chunks)} chunk(s):")
    for c in chunks:
        _info(f"  [{c['filename']}] chunk={c['chunk_index']} sim={round(c['similarity'], 3)}")
    _end()

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

class DecomposeInput(BaseModel):
    run_id: str = Field(description="The current run ID for context sharing")
    query:  str = Field(description="The user's original question to decompose")


@function_tool
async def query_decomposer_tool(input: DecomposeInput) -> str:
    """
    Breaks complex multi-part questions into focused sub-queries for better retrieval.
    Use only when the question is complex, multi-part, or ambiguous.
    Skip for simple focused questions.
    """
    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query analysis expert. Determine if the question needs "
                    "breaking into sub-questions for better document retrieval. "
                    "If simple, return as-is. If complex, break into 2-4 focused sub-questions. "
                    "Return ONLY the question(s), numbered if multiple."
                )
            },
            {"role": "user", "content": input.query}
        ],
        temperature=DECOMPOSER_TEMPERATURE,
        max_tokens=DECOMPOSER_MAX_TOKENS,
    )
    result = response.choices[0].message.content.strip()
    ctx = _run_context.get(input.run_id, {})
    ctx["decomposed_queries"] = result
    _run_context[input.run_id] = ctx
    print(f"[decomposer] => {result[:120]}")
    return result


# ── Tool 4: context_builder_tool ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str = Field(description="user or assistant")
    content: str = Field(description="Message content")


class ContextBuilderInput(BaseModel):
    run_id:       str               = Field(description="The current run ID for context sharing")
    user_query:   str               = Field(description="The user's original question")
    chat_history: list[ChatMessage] = Field(default=[], description="Previous conversation messages")


@function_tool
async def context_builder_tool(input: ContextBuilderInput) -> str:
    """
    Fuses vector search chunks AND knowledge graph results into one coherent context.
    Always call this after search_chunks_tool and/or graph_search_tool.
    Filters irrelevant content, resolves pronoun references from history,
    and incorporates entity relationship data from the graph.
    Call before answer_validator_tool.
    """
    ctx           = _run_context.get(input.run_id, {})
    chunks        = ctx.get("chunks", [])
    graph_results = ctx.get("graph_results", [])

    if not chunks and not graph_results:
        return "No results to build context from. Run search_chunks_tool or graph_search_tool first."

    _banner("context_builder_tool -- fusing sources")
    _info(f"vector chunks : {len(chunks)}")
    _info(f"graph results : {len(graph_results)}")
    if graph_results:
        _info("graph results included:")
        for r in graph_results:
            _info(f"  {r['file_name']} => {r.get('matched_entities', [])}")
    _end()

    chunk_section = "\n\n---\n\n".join([
        f"[Source: {c['filename']} | Chunk {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    ]) if chunks else "No vector chunks retrieved."

    graph_section = ""
    if graph_results:
        lines = ["Knowledge graph findings:"]
        for r in graph_results:
            matched = ", ".join(r.get("matched_entities", []))
            related = ", ".join(r.get("related_entities", [])[:5])
            lines.append(
                f"  File: {r['file_name']} | "
                f"Matched entities: {matched} | "
                f"Related entities: {related or 'none'}"
            )
        graph_section = "\n".join(lines)

    history_text = "\n".join([
        f"{m.role.upper()}: {m.content}" for m in input.chat_history[-CHAT_HISTORY_WINDOW:]
    ]) if input.chat_history else "None"

    combined = f"{chunk_section}\n\n{graph_section}".strip()

    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a context analyst. Given vector search chunks and knowledge "
                    "graph findings, your job is to:\n"
                    "1. Filter out content clearly irrelevant to the query\n"
                    "2. Resolve pronouns or references using chat history\n"
                    "3. Incorporate entity relationships from graph findings\n"
                    "4. Organize everything logically for answering the question\n"
                    "Keep all source labels intact. Do not add outside information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Query: {input.user_query}\n\n"
                    f"Chat History:\n{history_text}\n\n"
                    f"Retrieved Content:\n{combined}"
                )
            }
        ],
        temperature=CONTEXT_BUILDER_TEMPERATURE,
        max_tokens=CONTEXT_BUILDER_MAX_TOKENS,
    )

    enriched = response.choices[0].message.content.strip()
    ctx["enriched_context"] = enriched
    ctx["sources"] = [
        {
            "filename":    c["filename"],
            "chunk_index": c["chunk_index"],
            "similarity":  round(c["similarity"], 3),
            "excerpt":     c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
            "file_name":   (c.get("metadata") or {}).get("file_name"),
            "page":        (c.get("metadata") or {}).get("page"),
            "section":     (c.get("metadata") or {}).get("section"),
        }
        for c in chunks
    ]
    _run_context[input.run_id] = ctx
    return enriched


# ── Tool 5: answer_validator_tool ─────────────────────────────────────────────

class ValidatorInput(BaseModel):
    run_id: str = Field(description="The current run ID for context sharing")
    answer: str = Field(description="The proposed answer to validate")


@function_tool
async def answer_validator_tool(input: ValidatorInput) -> str:
    """
    Validates the proposed answer is grounded in retrieved context.
    Returns 'VALID' if fully supported, or lists unsupported claims.
    Always call this before returning the final answer.
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
                    "You are a fact-checking assistant. Check if every claim in the answer "
                    "is supported by the context. Reply 'VALID' if fully grounded. "
                    "Otherwise list each unsupported claim starting with '- UNSUPPORTED:'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nAnswer:\n{input.answer}"
            }
        ],
        temperature=VALIDATOR_TEMPERATURE,
        max_tokens=VALIDATOR_MAX_TOKENS,
    )
    result = response.choices[0].message.content.strip()
    verdict = "VALID" if result.strip() == "VALID" else "ISSUES FOUND"
    print(f"[validator] {verdict} -- {result[:100]}")
    return result


# ── Tool 6: graph_search_tool ─────────────────────────────────────────────────

class GraphSearchInput(BaseModel):
    run_id:     str = Field(description="The current run ID for context sharing")
    user_query: str = Field(description="The question or sub-query to search the knowledge graph for")
    max_hops:   int = Field(default=2, description="Relationship hops to traverse (1 or 2)")


async def _extract_query_entities(text: str) -> list[str]:
    """NER pass on query text -- returns entity name strings only."""
    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract named entities (people, companies, places, products, concepts) "
                    "from the user's question. Return ONLY a JSON object with key entities "
                    "containing an array of name strings. "
                    "Example: {\"entities\": [\"Acme Corp\", \"Manila\"]}. "
                    "Return {\"entities\": []} if no clear named entities exist."
                )
            },
            {"role": "user", "content": text[:500]},
        ],
        temperature=QUERY_NER_TEMPERATURE,
        max_tokens=QUERY_NER_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        return [str(x).strip() for x in parsed.get("entities", []) if x]
    except Exception:
        return []


@function_tool
async def graph_search_tool(input: GraphSearchInput) -> str:
    """
    Searches the Neo4j Aura knowledge graph for documents and entities related to the query.

    Use this when:
    - The question mentions specific people, companies, locations, or named concepts
    - The question asks about relationships between things
    - Vector search returned weak or no results for an entity-heavy question
    - The user asks about a specific named thing across documents

    Complements search_chunks_tool -- run both, then context_builder_tool fuses the results.
    """
    from utils.neo4j_client import graph_search

    ctx     = _run_context.get(input.run_id, {})
    user_id = ctx.get("user_id")

    if not user_id:
        return "Error: No user_id in context."

    _banner("NEO4J AURA HIT -- graph_search_tool")
    _info(f"run_id  : {input.run_id}")
    _info(f"user_id : {user_id}")
    _info(f"query   : {input.user_query[:80]}")

    entity_names = await _extract_query_entities(input.user_query)
    _info(f"NER entities : {entity_names}")

    if not entity_names:
        _warn("No named entities found -- skipping Cypher query")
        _end()
        return (
            "No named entities detected -- graph search needs a specific name to look up. "
            "Use search_chunks_tool for semantic search instead."
        )

    _info(f"Running Cypher on Neo4j Aura for: {entity_names}")
    results = await graph_search(
        user_id=user_id,
        entity_names=entity_names,
        max_hops=input.max_hops,
        limit=GRAPH_SEARCH_LIMIT,
    )

    if not results:
        _warn(f"Neo4j returned 0 documents for: {entity_names}")
        _end()
        return (
            f"No documents found in the knowledge graph for: {', '.join(entity_names)}. "
            "Try search_chunks_tool for a broader semantic search."
        )

    _ok(f"Neo4j returned {len(results)} document(s):")
    for r in results:
        matched = ", ".join(r.get("matched_entities", []))
        related = ", ".join(r.get("related_entities", [])[:5])
        _info(f"  {r['file_name']}  score={r.get('relevance_score', 0)}")
        _info(f"    matched : [{matched}]")
        _info(f"    related : [{related or 'none'}]")
    _end()

    ctx["graph_results"]  = results
    ctx["graph_entities"] = entity_names
    _run_context[input.run_id] = ctx

    lines = [f"Knowledge graph results for: {', '.join(entity_names)}\n"]
    for r in results:
        matched = ", ".join(r.get("matched_entities", []))
        related = ", ".join(r.get("related_entities", [])[:5])
        lines.append(
            f"File: {r['file_name']} | Score: {r.get('relevance_score', 0)}\n"
            f"  Matched entities : {matched}\n"
            f"  Related entities : {related or 'none'}\n"
        )
    return "\n".join(lines)


# ── Tool 7: entity_explorer_tool ──────────────────────────────────────────────

class EntityExploreInput(BaseModel):
    run_id:      str = Field(description="The current run ID for context sharing")
    entity_name: str = Field(description="Exact entity name to explore as it appears in documents")


@function_tool
async def entity_explorer_tool(input: EntityExploreInput) -> str:
    """
    Returns all entities directly connected to a named entity in the Neo4j knowledge graph.

    Use this when:
    - The user asks what is X connected to or tell me everything about X
    - You found an entity via graph_search_tool and want to explore its neighborhood
    - You want to discover related topics before running a targeted vector search

    Returns neighbor entities with their relationship types.
    """
    from utils.neo4j_client import get_entity_neighbors

    ctx     = _run_context.get(input.run_id, {})
    user_id = ctx.get("user_id")

    if not user_id:
        return "Error: No user_id in context."

    _banner("NEO4J AURA HIT -- entity_explorer_tool")
    _info(f"user_id     : {user_id}")
    _info(f"entity_name : {input.entity_name}")

    neighbors = await get_entity_neighbors(
        user_id=user_id,
        entity_name=input.entity_name,
        limit=ENTITY_EXPLORER_LIMIT,
    )

    if not neighbors:
        _warn(f"No connections found for '{input.entity_name}'")
        _end()
        return (
            f"No connections found for '{input.entity_name}'. "
            "It may not exist in the graph or may be stored under a slightly different name."
        )

    _ok(f"Neo4j returned {len(neighbors)} neighbor(s) for '{input.entity_name}':")
    for n in neighbors:
        _info(f"  {n['neighbor']} ({n['type']})  <->  {n['relation']}")
    _end()

    lines = [f"Connections for '{input.entity_name}':\n"]
    for n in neighbors:
        lines.append(f"  {n['neighbor']} ({n['type']})  <->  {n['relation']}")
    return "\n".join(lines)


# ── Tool 8: text2cypher_tool ──────────────────────────────────────────────────

# Graph schema description injected into the LLM prompt so it knows
# exactly what nodes, properties, and relationships exist.
# Update this if you add new labels or relationship types to your graph.
_GRAPH_SCHEMA = """
Node labels and properties:
  (:Document  {user_id: string, file_name: string, job_id: string})
  (:Entity    {user_id: string, name: string, type: string})
    type is one of: STUDENT, TEACHER, SCHOOL_HEAD, OFFICIAL,
                    SCHOOL, DISTRICT, DIVISION, REGION,
                    SUBJECT, GRADE_LEVEL, COMPETENCY, PROGRAM,
                    POLICY, PROVISION,
                    ASSESSMENT, METRIC,
                    PERIOD,
                    CONCEPT, CHARACTER, SETTING

Relationships:
  (:Entity)-[:APPEARS_IN {chunks: list}]->(:Document)
  (:Entity)-[:RELATES_TO {type: string}]->(:Entity)
    RELATES_TO.type examples: ISSUED_BY, APPLIES_TO, IMPLEMENTS, REFERENCES, SIGNED_BY,
      COVERS, PRESCRIBED_FOR, ALIGNED_WITH, USES_APPROACH, TAUGHT_IN,
      SCORED_ON, ENROLLED_IN, TEACHES, ASSESSED_IN, BELONGS_TO, ACHIEVED,
      IMPLEMENTED_BY, CONDUCTED_AT, SUPERVISED_BY, PARTICIPATED_IN,
      TEACHES_CONCEPT, SUITABLE_FOR, FEATURES

All nodes are scoped per user — every query MUST filter by user_id.
"""

_CYPHER_SYSTEM_PROMPT = f"""You are a Cypher query generator for a Neo4j knowledge graph.

Graph schema:
{_GRAPH_SCHEMA}

Rules you MUST follow — no exceptions:
1. ALWAYS include a user_id filter: {{user_id: $user_id}} on every node pattern
2. ALWAYS end with LIMIT $limit
3. Use only labels and relationship types defined in the schema above
4. Return only the raw Cypher query — no explanation, no markdown fences, no comments
5. If the question cannot be answered with the schema, return exactly: UNSUPPORTED

Good example:
  Question: "Which teachers are connected to Olongapo City National High School?"
  Cypher:
  MATCH (t:Entity {{user_id: $user_id, type: "TEACHER"}})-[:RELATES_TO]-(s:Entity {{user_id: $user_id, name: "Olongapo City National High School"}})
  RETURN t.name AS teacher, s.name AS school
  LIMIT $limit

Bad example (NEVER do this — missing user_id):
  MATCH (e:Entity) WHERE e.name = "Olongapo City National High School" RETURN e
"""


class Text2CypherInput(BaseModel):
    run_id:   str = Field(description="The current run ID for context sharing")
    question: str = Field(description="Natural language question to convert to a Cypher query")
    limit:    int = Field(default=TEXT2CYPHER_DEFAULT_LIMIT, description="Max rows to return from Neo4j")


@function_tool
async def text2cypher_tool(input: Text2CypherInput) -> str:
    """
    Converts a natural language question into a Cypher query and runs it against Neo4j Aura.

    Use this for COMPLEX graph traversals that graph_search_tool cannot handle, such as:
    - Multi-hop relationship questions: "which policies apply to Grade 4 within 2 hops?"
    - Aggregation questions: "which entity appears in the most documents?"
    - Path questions: "what is the relationship chain between a teacher and a school?"
    - Filtered traversals: "find all SCHOOL entities that appear in more than one document"

    Do NOT use for simple entity lookups — use graph_search_tool for those.
    Always call context_builder_tool after this to fuse results with vector chunks.
    """
    from utils.neo4j_client import get_neo4j

    ctx     = _run_context.get(input.run_id, {})
    user_id = ctx.get("user_id")

    if not user_id:
        return "Error: No user_id in context."

    limit = max(1, min(TEXT2CYPHER_MAX_LIMIT, input.limit))

    _banner("NEO4J AURA HIT -- text2cypher_tool")
    _info(f"user_id  : {user_id}")
    _info(f"question : {input.question}")

    # ── Step 1: LLM generates the Cypher ─────────────────────────────────────
    response = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _CYPHER_SYSTEM_PROMPT},
            {"role": "user",   "content": input.question},
        ],
        temperature=TEXT2CYPHER_TEMPERATURE,
        max_tokens=TEXT2CYPHER_MAX_TOKENS,
    )
    raw_cypher = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw_cypher.startswith("```"):
        raw_cypher = "\n".join(
            line for line in raw_cypher.splitlines()
            if not line.startswith("```")
        ).strip()

    _info(f"generated Cypher:\n{raw_cypher}")

    if raw_cypher.upper() == "UNSUPPORTED":
        _warn("LLM flagged question as unsupported by the schema")
        _end()
        return (
            "This question cannot be answered by a graph traversal with the current schema. "
            "Try graph_search_tool or search_chunks_tool instead."
        )

    # ── Step 2: Safety check — block queries missing user_id filter ───────────
    # This is the critical multi-tenant guard. If the LLM forgot to include
    # $user_id in the generated Cypher, we refuse to run it.
    if "$user_id" not in raw_cypher:
        _warn("SAFETY BLOCK — generated Cypher missing $user_id filter, refusing to run")
        _end()
        return (
            "Generated Cypher was missing the required user_id filter and was blocked for safety. "
            "Try rephrasing your question or use graph_search_tool instead."
        )

    # ── Step 3: Execute against Neo4j Aura ───────────────────────────────────
    try:
        driver = get_neo4j()
        async with driver.session() as session:
            result = await session.run(
                raw_cypher,
                user_id=user_id,   # always injected — never trust LLM to embed it
                limit=limit,
            )
            rows = await result.data()

    except Exception as e:
        _warn(f"Cypher execution failed: {e}")
        _end()
        return (
            f"The generated Cypher query failed to execute: {e}\n"
            "Try rephrasing your question or use graph_search_tool instead."
        )

    if not rows:
        _warn("Neo4j returned 0 rows")
        _end()
        return (
            "The graph query returned no results. "
            "The entities or relationships in your question may not exist in the graph yet."
        )

    _ok(f"Neo4j returned {len(rows)} row(s):")
    for row in rows[:5]:
        _info(f"  {row}")
    if len(rows) > 5:
        _info(f"  ... and {len(rows) - 5} more")
    _end()

    # Store graph results so context_builder_tool can include them
    ctx["graph_results"] = ctx.get("graph_results", []) + [
        {"file_name": "[graph query]", "matched_entities": list(row.values()), "related_entities": []}
        for row in rows[:8]
    ]
    _run_context[input.run_id] = ctx

    # Format rows as readable table
    if not rows:
        return "No results."

    headers = list(rows[0].keys())
    lines   = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))

    return f"Graph query results ({len(rows)} rows):\n" + "\n".join(lines)