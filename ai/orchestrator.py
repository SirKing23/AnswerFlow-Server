import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Optional
from agents import Agent, Runner, ModelSettings

# All 8 tools live in tools.py
from ai.tools import (
    user_query_embedder_tool,
    search_chunks_tool,
    query_decomposer_tool,
    context_builder_tool,
    answer_validator_tool,
    graph_search_tool,
    entity_explorer_tool,
    text2cypher_tool,
    set_run_context,
    get_run_context,
    clear_run_context,
)

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
import os as _os
_os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


ORCHESTRATOR_INSTRUCTIONS = """
You are an intelligent RAG orchestrator that answers questions using the user's uploaded documents.
You have access to TWO retrieval systems — use both when appropriate:

  1. Vector search  (pgvector / Supabase) — semantic similarity
  2. Knowledge graph (Neo4j Aura)         — entity & relationship lookup

━━━ TOOLS ━━━

1. query_decomposer_tool
   Use ONLY when the question is complex or multi-part. Skip for simple questions.

2. user_query_embedder_tool
   Always call before search_chunks_tool to create the vector embedding.

3. search_chunks_tool
   Semantic similarity search across the user's documents.
   Call after embedding. Use for any content-level question.

4. graph_search_tool
   Searches Neo4j for documents containing specific named entities.
   Call when the question mentions a specific person, company, place, product, or concept.
   Call when asking about relationships between things.
   Call when vector search returns weak results on entity-heavy questions.

5. entity_explorer_tool
   Explores all direct connections of a named entity in the knowledge graph.
   Call after graph_search_tool when you want to dig deeper into one entity's neighborhood.

6. text2cypher_tool
   Converts a natural language question into a Cypher query and runs it on Neo4j.
   Use for COMPLEX graph traversals that graph_search_tool cannot handle:
     - Multi-hop: "who are all people connected to Acme Corp within 2 hops?"
     - Aggregation: "which entity appears in the most documents?"
     - Path finding: "what connects John to Manila?"
     - Filtered: "find all ORG entities appearing in more than one document"
   Do NOT use for simple entity lookups — use graph_search_tool for those.

7. context_builder_tool
   ALWAYS call after ALL searches are complete (vector + graph).
   Fuses all result types into a coherent context for answering.

8. answer_validator_tool
   ALWAYS call before your final answer. Catches hallucinations.
   Revise and re-validate if it returns UNSUPPORTED claims.

━━━ DECISION GUIDE ━━━

Simple factual question:
  → embed → search_chunks_tool → context_builder_tool → validate → answer

Entity question ("docs about Acme Corp", "what did John say about X"):
  → graph_search_tool + embed + search_chunks_tool → context_builder_tool → validate → answer

Relationship question ("how does X relate to Y", "what connects A and B"):
  → graph_search_tool → entity_explorer_tool → context_builder_tool → validate → answer

Complex graph traversal ("who is connected to X within 2 hops", "which entity appears most"):
  → text2cypher_tool → context_builder_tool → validate → answer

Complex multi-part question:
  → decompose → for each sub-query: appropriate search tools → context_builder_tool → validate → answer

━━━ RULES ━━━
- Answer ONLY from retrieved document content. Never use outside knowledge.
- Always cite which source file your answer comes from.
- If no search returns results, say so clearly — do not make up an answer.
- Be concise and direct. Use bullet points for lists.
- The run_id is provided in the first user message — pass it to every tool call.
"""


async def run_agent(
    user_id:  str,
    message:  str,
    history:  list,
    file_id:  Optional[str],
    run_id:   Optional[str] = None,
) -> dict:
    """
    Run the agentic RAG pipeline for a single user message.
    The agent decides which tools to call — vector search, graph search, or both.
    All tools share state via run_id stored in _run_context.
    """
    if not run_id:
        run_id = str(uuid.uuid4())

    set_run_context(run_id, user_id, file_id)

    agent = Agent(
        name="RAG Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=OPENAI_CHAT_MODEL,
        model_settings=ModelSettings(
            parallel_tool_calls=False,
            temperature=0.7,
            presence_penalty=0.3,
            max_tokens=2048,
        ),
        tools=[
            query_decomposer_tool,
            user_query_embedder_tool,
            search_chunks_tool,
            graph_search_tool,        # Neo4j entity search
            entity_explorer_tool,     # Neo4j neighborhood exploration
            text2cypher_tool,         # Neo4j natural language Cypher
            context_builder_tool,
            answer_validator_tool,
        ],
    )

    history_text = ""
    if history:
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-6:]
        ])
        history_text = f"\n\nConversation history:\n{history_text}"

    agent_input = (
        f"run_id: {run_id}\n"
        f"user_id: {user_id}\n"
        f"file_id: {file_id or 'None (search all files)'}\n"
        f"{history_text}\n\n"
        f"User question: {message}"
    )

    try:
        result = await Runner.run(agent, agent_input, max_turns=25)
        ctx     = get_run_context(run_id)
        return {
            "answer":  result.final_output,
            "sources": ctx.get("sources", []),
            "run_id":  run_id,
            "error":   None,
        }
    except Exception as e:
        return {
            "answer":  None,
            "sources": [],
            "run_id":  run_id,
            "error":   str(e),
        }
    finally:
        clear_run_context(run_id)