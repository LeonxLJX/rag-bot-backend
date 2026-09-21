"""
RAG Bot Backend — Main FastAPI Application

Enterprise-grade RAG platform with:
- Multi-tenant knowledge bases
- Document upload & ingestion
- Hybrid Search (Vector + BM25)
- Reranking
- Conversation memory
- Agent tools
- Guardrails (anti-hallucination)
- REST API with CORS
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from config.settings import settings
from api.routes import chat, knowledge_base
from core.knowledge_base.manager import KnowledgeBaseManager


# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Bot Backend",
    description="""
    Enterprise-grade RAG platform with multi-tenant knowledge bases.
    
    Features:
    - Multi-tenant knowledge base isolation
    - Hybrid Search (Vector + BM25)
    - Document ingestion (PDF/TXT/MD/DOCX)
    - Reranking (Cross-Encoder)
    - Conversation memory
    - Agent tools
    - Guardrails (anti-hallucination)
    """,
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Managers ──────────────────────────────────────────────────────────

kb_manager = KnowledgeBaseManager()


# ─── Routes ────────────────────────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(knowledge_base.router)


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "hybrid_search": settings.USE_HYBRID_SEARCH,
            "reranker": settings.USE_RERANKER,
            "guardrail": settings.USE_GUARDRAIL,
            "cache": settings.USE_CACHE,
            "conversation_memory": True,
            "agent_tools": True,
        },
    }


# ─── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RAG Bot Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Multi-tenant knowledge bases",
            "Hybrid Search (Vector + BM25)",
            "Document ingestion (PDF/TXT/MD/DOCX)",
            "Reranking (Cross-Encoder)",
            "Conversation memory",
            "Agent tools",
            "Guardrails (anti-hallucination)",
        ],
    }


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Create data directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/vector_db", exist_ok=True)

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
