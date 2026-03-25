import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Optional
from agents import Agent, Runner, ModelSettings
from ai.tools import (
    user_query_embedder_tool,
    search_chunks_tool,
    query_decomposer_tool,
    context_builder_tool,
    answer_validator_tool,
    set_run_context,
    get_run_context,
    clear_run_context,
)
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
import os as _os
_os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


ORCHESTRATOR_INSTRUCTIONS = """
You are an intelligent RAG (Retrieval-Augmented Generation) orchestrator.
Your job is to answer the user's question using ONLY their uploaded documents.
If the question is complex, decompose it into simpler sub-questions.
You can create a plan for which tools to call and in what order, but you MUST use the tools provided.
You can reconstruct the user's query if needed, if you think it will help the embedding and retrieval process, but you cannot use any external knowledge or call any tools other than the ones provided.

You have access to these tools — use them intelligently:

1. query_decomposer_tool
   Use ONLY when the question is complex, multi-part, or ambiguous.
   Skip for simple focused questions.

2. user_query_embedder_tool
   Call this to embed the query (or each sub-query if decomposed).
   Skip if the query is simple and focused.
   Pass the run_id and the text to embed.

3. search_chunks_tool
   ALWAYS call this after embedding to retrieve relevant document chunks.
   You can call it multiple times with different similarity thresholds if needed.

4. context_builder_tool
   ALWAYS call this after search to analyze and structure the retrieved context.
   Pass the run_id, the original user_query, and chat_history.

5. answer_validator_tool
   ALWAYS call this before giving your final answer to check for hallucinations.
   If it returns unsupported claims, revise your answer and validate again.

Rules:
- Base your answer ONLY on retrieved document content. Never use outside knowledge.
- Always cite which source file your answer comes from.
- If no relevant chunks are found, say so clearly — do not make up an answer.
- Be concise and direct. Use bullet points for lists.
- The run_id for this session will be provided in the first user message.
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
    The agent decides which tools to call based on the query.
    All tools share state via run_id stored in _run_context.
    """
    if not run_id:
        run_id = str(uuid.uuid4())

    set_run_context(run_id, user_id, file_id)

    agent = Agent(
        name="RAG Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=OPENAI_CHAT_MODEL,
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=[
            query_decomposer_tool,
            user_query_embedder_tool,
            search_chunks_tool,
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
        result = await Runner.run(agent, agent_input)
        final_answer = result.final_output
        ctx     = get_run_context(run_id)
        sources = ctx.get("sources", [])

        return {
            "answer":  final_answer,
            "sources": sources,
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
