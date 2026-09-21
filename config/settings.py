"""
RAG Bot Backend — Enterprise RAG Platform
Inspired by Langchain-Chatchat, simplified for production use.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./data/ragbot.db"

    # Vector Store
    VECTOR_DB_PATH: str = "./data/vector_db"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1000

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    RETRIEVER_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    USE_HYBRID_SEARCH: bool = True
    USE_RERANKER: bool = True
    USE_GUARDRAIL: bool = True
    USE_CACHE: bool = True

    # Auth
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # Multi-tenant
    DEFAULT_KB_ID: str = "default"

    class Config:
        env_file = ".env"


settings = Settings()
