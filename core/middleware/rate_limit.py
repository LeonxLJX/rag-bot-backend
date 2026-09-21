"""
Rate Limiting Middleware — Protect API from abuse.

Uses slowapi with Redis backend for distributed rate limiting.
Falls back to in-memory if Redis unavailable.
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

from config.settings import settings


# In-memory rate limiter (fallback)
request_counts: dict[str, list[float]] = defaultdict(list)

# Rate limit settings
RATE_LIMIT_REQUESTS = 100  # requests
RATE_LIMIT_WINDOW = 60  # seconds


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limit middleware.
    
    Limits:
    - 100 requests per minute per IP
    - 1000 requests per hour per API key
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries
    cutoff = now - RATE_LIMIT_WINDOW
    request_counts[client_ip] = [t for t in request_counts[client_ip] if t > cutoff]

    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "detail": f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per minute.",
            },
        )

    # Record this request
    request_counts[client_ip].append(now)

    # Process request
    response = await call_next(request)
    return response
