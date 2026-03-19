import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from middleware.auth import verify_jwt
from services.pipelineChat import chat_with_docs

router = APIRouter()


class Message(BaseModel):
    role: str    # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []       # previous messages in the conversation
    file_id: Optional[str] = None     # if set, search only this file
                                      # if None, search all user's files


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]               # which chunks were used


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    """
    RAG chat endpoint.

    Accepts:
      - message:  the user's current question
      - history:  list of previous { role, content } messages
      - file_id:  optional — scope search to one file, or None for all files

    Returns:
      - answer:   LLM response grounded in the user's documents
      - sources:  list of chunks used, with file name and chunk index
    """
    # Verify JWT and extract user_id
    user_id = await verify_jwt(request)

    result = await chat_with_docs(
        user_id=user_id,
        message=body.message,
        history=body.history,
        file_id=body.file_id,
    )

    return ChatResponse(**result)
