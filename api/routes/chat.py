"""
Chat Routes — Conversation endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from core.rag.engine import RAGEngine
from config.settings import settings

router = APIRouter(prefix="/api/chat", tags=["chat"])


# In-memory RAG engines per knowledge base
rag_engines: dict[str, RAGEngine] = {}


def get_rag_engine(kb_id: str = None) -> RAGEngine:
    """Get or create RAG engine for a knowledge base."""
    kb_id = kb_id or settings.DEFAULT_KB_ID
    if kb_id not in rag_engines:
        rag_engines[kb_id] = RAGEngine(kb_id=kb_id)
    return rag_engines[kb_id]


class ChatRequest(BaseModel):
    message: str
    kb_id: Optional[str] = None
    session_id: Optional[str] = None


class Source(BaseModel):
    content: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    kb_id: str
    session_id: Optional[str] = None


@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to the RAG chatbot.
    
    - **message**: User's question
    - **kb_id**: Knowledge base ID (default: default)
    - **session_id**: Chat session ID for multi-turn
    """
    engine = get_rag_engine(req.kb_id)

    try:
        result = engine.query(req.message)
    except ValueError as e:
        # No documents loaded yet
        return ChatResponse(
            answer="No documents loaded in this knowledge base yet. Please upload documents first.",
            sources=[],
            kb_id=req.kb_id or settings.DEFAULT_KB_ID,
            session_id=req.session_id,
        )

    return ChatResponse(
        answer=result["answer"],
        sources=[],  # TODO: return actual sources
        kb_id=result["kb_id"],
        session_id=req.session_id,
    )


@router.get("/history/{session_id}")
def get_chat_history(session_id: str):
    """Get chat history for a session."""
    # TODO: implement with database
    return {"session_id": session_id, "messages": []}
