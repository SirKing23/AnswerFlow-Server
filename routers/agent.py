import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from middleware.auth import verify_jwt
from ai.orchestrator import run_agent

router = APIRouter()


class Message(BaseModel):
    role:    str
    content: str


class AgentRequest(BaseModel):
    message:  str
    history:  list[Message] = []
    file_id:  Optional[str] = None
    run_id:   Optional[str] = None   # pass back from previous response for continuity


class AgentResponse(BaseModel):
    answer:  str
    sources: list[dict]
    run_id:  str                      # frontend stores this and sends back next turn


@router.post("/api/agent")
async def agent_chat(request: Request, body: AgentRequest):
    """
    Agentic RAG endpoint.

    Unlike /api/chat which always runs a fixed pipeline,
    this endpoint uses an AI orchestrator that decides which
    tools to call based on the complexity of the query.

    Tool flow (agent decides):
      query_decomposer_tool  → breaks complex questions apart
      user_query_embedder_tool → embeds the query
      search_chunks_tool       → retrieves relevant chunks
      context_builder_tool     → enriches and filters context
      answer_validator_tool    → checks answer is grounded

    Returns run_id — store it on the frontend and send back
    with each message for conversation continuity.
    """
    user_id = await verify_jwt(request)

    history = [{"role": m.role, "content": m.content} for m in body.history]
  

    result = await run_agent(
        user_id=user_id,
        message=body.message,
        history=history,
        file_id=body.file_id,
        run_id=body.run_id,
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return AgentResponse(
        answer=result["answer"],
        sources=result["sources"],
        run_id=result["run_id"],
    )
