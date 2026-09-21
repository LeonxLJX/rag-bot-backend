"""
RAG Bot Backend — Production Settings

Supports both:
- Development: SQLite + Chroma + in-memory cache
- Production: PostgreSQL + Pinecone + Redis + Celery

Switch via environment variables in .env
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # ─── API ─────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # ─── Database ─────────────────────────────────────────────────────────
    # Dev: sqlite:///./data/ragbot.db
    # Prod: postgresql://user:pass@host:5432/ragbot
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/ragbot.db"
    )

    # ─── Vector Store ────────────────────────────────────────────────────
    # Dev: chroma
    # Prod: pinecone
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    VECTOR_DB_PATH: str = "./data/vector_db"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536

    # Pinecone (production)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ragbot")

    # ─── LLM ─────────────────────────────────────────────────────────────
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1000

    # ─── RAG ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    RETRIEVER_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    USE_HYBRID_SEARCH: bool = True
    USE_RERANKER: bool = True
    USE_GUARDRAIL: bool = True
    USE_CACHE: bool = True

    # ─── Auth ────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # ─── Redis (Production Cache) ──────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = 3600

    # ─── Celery (Production Async Tasks) ─────────────────────────────────
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

    # ─── Monitoring ──────────────────────────────────────────────────────
    # Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in production
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

    # ─── Multi-tenant ────────────────────────────────────────────────────
    DEFAULT_KB_ID: str = "default"

    class Config:
        env_file = ".env"


settings = Settings()
