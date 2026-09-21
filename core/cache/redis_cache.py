"""
Cache Layer — Redis caching for RAG responses.

Features:
- Cache frequent queries
- TTL-based expiration
- Cache invalidation on document update
- Fallback to in-memory cache if Redis unavailable
"""
import json
import hashlib
from typing import Optional, Any
from datetime import timedelta

from config.settings import settings


class Cache:
    """
    Cache layer with Redis backend.
    
    Falls back to in-memory dict if Redis is not available.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._memory_cache: dict = {}
        self._redis = None
        self._connect_redis()

    def _connect_redis(self):
        """Try to connect to Redis."""
        try:
            import redis
            self._redis = redis.Redis(host="localhost", port=6379, db=0)
            self._redis.ping()
            print("✅ Connected to Redis")
        except Exception:
            print("⚠️ Redis not available, using in-memory cache")
            self._redis = None

    def _make_key(self, query: str, kb_id: str) -> str:
        """Create a cache key from query and KB ID."""
        raw = f"{kb_id}:{query}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, kb_id: str) -> Optional[Any]:
        """Get cached response."""
        key = self._make_key(query, kb_id)

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        else:
            if key in self._memory_cache:
                return self._memory_cache[key]

        return None

    def set(self, query: str, kb_id: str, value: Any):
        """Cache a response."""
        key = self._make_key(query, kb_id)
        data = json.dumps(value)

        if self._redis:
            try:
                self._redis.setex(key, self.ttl, data)
            except Exception:
                pass
        else:
            self._memory_cache[key] = value

    def invalidate_kb(self, kb_id: str):
        """Invalidate all cache for a knowledge base."""
        if self._redis:
            try:
                # Delete all keys matching kb_id pattern
                pattern = f"*:{kb_id}:*"
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
            except Exception:
                pass
        else:
            # Clear memory cache for this KB
            keys_to_delete = [k for k in self._memory_cache if kb_id in k]
            for k in keys_to_delete:
                del self._memory_cache[k]

    def clear(self):
        """Clear all cache."""
        if self._redis:
            try:
                self._redis.flushdb()
            except Exception:
                pass
        else:
            self._memory_cache.clear()


# Global cache instance
cache = Cache(ttl_seconds=3600)
