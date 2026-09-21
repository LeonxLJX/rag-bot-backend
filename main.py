"""
RAG Bot Backend — Production-Grade FastAPI Application

Enterprise RAG platform with:
- Multi-tenant knowledge bases
- Hybrid Search (Vector + BM25)
- Reranking (Cross-Encoder)
- JWT Authentication
- SQLAlchemy ORM
- Redis Caching
- Async Tasks (Celery-ready)
- Rate Limiting
- Structured Logging
- Metrics & Monitoring
- Docker Support
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os

from config.settings import settings
from api.routes import chat, knowledge_base
from core.knowledge_base.manager import KnowledgeBaseManager
from core.middleware.rate_limit import rate_limit_middleware
from core.logging.structured_logger import logger
from core.monitoring.metrics import metrics_collector


# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Bot Backend",
    description="""
    Production-grade RAG platform with multi-tenant knowledge bases.
    
    Features:
    - Multi-tenant knowledge base isolation
    - Hybrid Search (Vector + BM25)
    - Reranking (Cross-Encoder)
    - JWT Authentication
    - Redis Caching
    - Async Task Processing
    - Rate Limiting
    - Structured Logging
    - Metrics & Monitoring
    """,
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def logging_and_metrics(request: Request, call_next):
    """Log every request and collect metrics."""
    start_time = time.time()

    # Rate limit check
    response = await rate_limit_middleware(request, call_next)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Collect metrics
    endpoint = request.url.path
    metrics_collector.record_request(
        endpoint=endpoint,
        latency_ms=latency_ms,
        error=response.status_code >= 400,
    )

    # Structured log
    logger.info(
        f"{request.method} {endpoint}",
        extra={
            "endpoint": endpoint,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return response


# ─── Global Managers ──────────────────────────────────────────────────────────

kb_manager = KnowledgeBaseManager()


# ─── Routes ────────────────────────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(knowledge_base.router)


# ─── Health & Metrics ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "hybrid_search": settings.USE_HYBRID_SEARCH,
            "reranker": settings.USE_RERANKER,
            "guardrail": settings.USE_GUARDRAIL,
            "cache": settings.USE_CACHE,
            "rate_limiting": True,
            "structured_logging": True,
            "metrics": True,
        },
    }


@app.get("/metrics")
def get_metrics():
    """Get system metrics."""
    return metrics_collector.get_stats()


# ─── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RAG Bot Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Create data directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/vector_db", exist_ok=True)

    logger.info("Starting RAG Bot Backend", extra={"port": settings.API_PORT})

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
