"""
Chat Routes — Conversation endpoints.

Real retrieval contexts are returned as `sources` (no more empty placeholders),
and multi-turn history is persisted in a process-local session store:
in-memory dict, capped per session, TTL-evicted on read. Production should
swap this for Redis (same interface, see _SESSIONS comments).
"""
import time
import uuid
from fastapi import APIRouter
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


# ─── Session store (process-local; production: Redis with the same shape) ─────

MAX_MESSAGES_PER_SESSION = 50
SESSION_TTL_SECONDS = 24 * 3600

# session_id -> {"messages": [...], "last_seen": epoch_seconds}
_SESSIONS: dict[str, dict] = {}


def _append_message(session_id: str, role: str, content: str) -> None:
    """Record one turn, trimming to the newest MAX_MESSAGES_PER_SESSION."""
    entry = _SESSIONS.setdefault(session_id, {"messages": [], "last_seen": time.time()})
    entry["last_seen"] = time.time()
    entry["messages"].append({"role": role, "content": content, "ts": entry["last_seen"]})
    if len(entry["messages"]) > MAX_MESSAGES_PER_SESSION:
        entry["messages"] = entry["messages"][-MAX_MESSAGES_PER_SESSION:]


def _evict_expired() -> None:
    now = time.time()
    expired = [sid for sid, e in _SESSIONS.items() if now - e["last_seen"] > SESSION_TTL_SECONDS]
    for sid in expired:
        _SESSIONS.pop(sid, None)


@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to the RAG chatbot.

    - **message**: User's question
    - **kb_id**: Knowledge base ID (default: default)
    - **session_id**: Chat session ID for multi-turn; created when omitted
    """
    engine = get_rag_engine(req.kb_id)
    session_id = req.session_id or uuid.uuid4().hex

    try:
        result = engine.query(req.message)
    except ValueError:
        # No documents loaded yet — still record the turn so history is honest.
        _append_message(session_id, "user", req.message)
        return ChatResponse(
            answer="No documents loaded in this knowledge base yet. Please upload documents first.",
            sources=[],
            kb_id=req.kb_id or settings.DEFAULT_KB_ID,
            session_id=session_id,
        )

    sources = [Source(content=c["content"], score=c.get("score")) for c in result.get("contexts", [])]

    _append_message(session_id, "user", req.message)
    _append_message(session_id, "assistant", result["answer"])

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        kb_id=result["kb_id"],
        session_id=session_id,
    )


@router.get("/history/{session_id}")
def get_chat_history(session_id: str):
    """Return the retained messages for a session (TTL-evicted, capped)."""
    _evict_expired()
    entry = _SESSIONS.get(session_id)
    return {
        "session_id": session_id,
        "messages": entry["messages"] if entry else [],
    }
