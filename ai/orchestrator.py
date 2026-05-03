import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Optional
from agents import Agent, Runner, ModelSettings

# All tools live in tools.py
from ai.tools import (
    vector_search_tool,
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

from config import (
    OPENAI_API_KEY, OPENAI_CHAT_MODEL,
    ORCHESTRATOR_TEMPERATURE, ORCHESTRATOR_PRESENCE_PENALTY,
    ORCHESTRATOR_MAX_TOKENS, ORCHESTRATOR_MAX_TURNS,
    CHAT_HISTORY_WINDOW,
    ORCHESTRATOR_INSTRUCTIONS,  # agent instructions — defined in .env
)
import os as _os
_os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ORCHESTRATOR_INSTRUCTIONS is loaded from .env via config.py.
# It defines the agent's persona, tool usage guide, decision rules, and anti-loop rules.
# Edit the ORCHESTRATOR_INSTRUCTIONS key in .env to tune agent behaviour.


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
            parallel_tool_calls=True,
            temperature=ORCHESTRATOR_TEMPERATURE,
            presence_penalty=ORCHESTRATOR_PRESENCE_PENALTY,
            max_tokens=ORCHESTRATOR_MAX_TOKENS,
        ),
        tools=[
            query_decomposer_tool,
            vector_search_tool,       # embed + pgvector search in one step
            graph_search_tool,        # Neo4j entity search
            entity_explorer_tool,     # Neo4j neighborhood exploration
            text2cypher_tool,         # Neo4j natural language Cypher
            context_builder_tool,
            answer_validator_tool
        ],
    )

    history_text = ""
    if history:
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-CHAT_HISTORY_WINDOW:]
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
        result = await Runner.run(agent, agent_input, max_turns=ORCHESTRATOR_MAX_TURNS)
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