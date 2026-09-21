"""
Unit Tests — RAG Bot Backend

Tests:
- Config validation
- Document ingestion
- RAG engine (mocked)
- Auth (JWT)
- Cache
"""
import pytest
from config.settings import Settings


class TestConfig:
    """Test configuration."""

    def test_default_settings(self):
        """Test default settings are loaded."""
        settings = Settings()
        assert settings.API_PORT == 8000
        assert settings.CHUNK_SIZE == 800
        assert settings.CHUNK_OVERLAP == 100
        assert settings.RETRIEVER_TOP_K == 20
        assert settings.RERANK_TOP_N == 5

    def test_rag_settings(self):
        """Test RAG settings."""
        settings = Settings()
        assert settings.USE_HYBRID_SEARCH is True
        assert settings.USE_RERANKER is True
        assert settings.USE_GUARDRAIL is True
        assert settings.LLM_TEMPERATURE == 0.2


class TestDocumentIngestion:
    """Test document ingestion."""

    def test_splitter(self):
        """Test text splitter."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

        text = "This is a test document. " * 50
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(c) <= 150 for c in chunks)


class TestAuth:
    """Test authentication."""

    def test_password_hashing(self):
        """Test password hashing."""
        from core.auth.jwt_auth import get_password_hash, verify_password

        password = "testpassword123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    def test_jwt_token(self):
        """Test JWT token creation and decoding."""
        from core.auth.jwt_auth import create_access_token, decode_access_token

        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)

        assert token is not None
        decoded = decode_access_token(token)
        assert decoded["sub"] == "testuser"
        assert decoded["user_id"] == 1


class TestCache:
    """Test cache layer."""

    def test_memory_cache(self):
        """Test in-memory cache."""
        from core.cache.redis_cache import Cache

        cache = Cache(ttl_seconds=60)

        # Set and get
        cache.set("test query", "kb1", {"answer": "test"})
        result = cache.get("test query", "kb1")

        assert result is not None
        assert result["answer"] == "test"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from core.cache.redis_cache import Cache

        cache = Cache(ttl_seconds=60)
        result = cache.get("nonexistent query", "kb1")
        assert result is None


class TestKnowledgeBaseManager:
    """Test knowledge base manager."""

    def test_create_kb(self):
        """Test KB creation."""
        from core.knowledge_base.manager import KnowledgeBaseManager

        manager = KnowledgeBaseManager(base_dir="/tmp/test_kb")
        kb = manager.create_kb("test_kb", "Test KB", "Test description")

        assert kb.kb_id == "test_kb"
        assert kb.name == "Test KB"

    def test_list_kbs(self):
        """Test listing KBs."""
        from core.knowledge_base.manager import KnowledgeBaseManager

        manager = KnowledgeBaseManager(base_dir="/tmp/test_kb2")
        manager.create_kb("kb1", "KB 1")
        manager.create_kb("kb2", "KB 2")

        kbs = manager.list_kbs()
        assert len(kbs) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
