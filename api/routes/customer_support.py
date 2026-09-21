"""
Customer Support RAG — Real-world customer support chatbot.

Scenario: A company has 12,000 support articles.
Goal: Reduce 40% of repetitive tickets by letting customers self-serve.

API Endpoints:
- POST /api/support/chat — Customer asks a question
- POST /api/support/upload — Upload a support article (admin)
- GET /api/support/articles — List all articles (admin)
- GET /api/support/stats — Usage stats (admin)
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

from core.file_rag.engine import FileRAGEngine
from core.knowledge_base.manager import KnowledgeBaseManager
from core.auth.jwt_auth import get_current_user
from core.cache.redis_cache import cache
from core.logging.structured_logger import logger


router = APIRouter(prefix="/api/support", tags=["customer-support"])


# ─── Data Models ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    customer_email: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    confidence: float
    needs_human: bool = False


# ─── Global State ─────────────────────────────────────────────────────────────

# Support KB (default knowledge base for customer support)
SUPPORT_KB_ID = "customer-support"
support_engine = None
kb_manager = KnowledgeBaseManager()


def get_support_engine() -> FileRAGEngine:
    """Get or create the support RAG engine."""
    global support_engine
    if support_engine is None:
        support_engine = FileRAGEngine(kb_id=SUPPORT_KB_ID)
    return support_engine


# ─── Chat Endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Customer support chatbot.
    
    Flow:
    1. Check cache
    2. RAG retrieve + rerank + generate
    3. Guardrail check
    4. Return answer + sources
    5. If low confidence → needs_human=True
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Check cache
    cached = cache.get(request.question, SUPPORT_KB_ID)
    if cached:
        logger.info("Cache hit", extra={"question": request.question})
        return ChatResponse(
            **cached,
            session_id=session_id,
        )

    # Get RAG engine
    engine = get_support_engine()

    # Query
    result = engine.query(request.question)

    # Determine if needs human
    confidence = result.get("avg_score", 0.5)
    needs_human = confidence < 0.3 or "I don't have enough information" in result["answer"]

    response_data = {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": confidence,
        "needs_human": needs_human,
    }

    # Cache the response
    cache.set(request.question, SUPPORT_KB_ID, response_data)

    logger.info(
        "Chat request",
        extra={
            "question": request.question[:50],
            "confidence": confidence,
            "needs_human": needs_human,
        },
    )

    return ChatResponse(
        **response_data,
        session_id=session_id,
    )


# ─── Article Management ───────────────────────────────────────────────────────

@router.post("/upload")
async def upload_article(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a support article (admin only)."""
    # Save file
    file_path = f"data/uploads/support/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Invalidate cache for this KB
    cache.invalidate_kb(SUPPORT_KB_ID)

    # Add to RAG engine (async in production)
    # engine.add_document(file_path)

    return {
        "message": "Article uploaded successfully",
        "filename": file.filename,
        "cached_invalidated": True,
    }


@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Get support KB stats."""
    engine = get_support_engine()
    return {
        "kb_id": SUPPORT_KB_ID,
        "num_chunks": len(engine.chunks),
        "retriever_type": "hybrid+rerank",
        "guardrail_enabled": True,
    }


# ─── Example Knowledge Base Content ────────────────────────────────────────────

SAMPLE_SUPPORT_ARTICLES = [
    {
        "title": "How to reset your password",
        "content": """To reset your password:
1. Go to the login page
2. Click "Forgot password"
3. Enter your email address
4. Check your email for reset link
5. Click the link and enter new password
6. Password must be at least 8 characters with one number and one special character

If you don't receive the email within 5 minutes, check your spam folder."""
    },
    {
        "title": "Refund policy",
        "content": """Our refund policy:
- Full refund within 30 days of purchase
- No questions asked
- Refund processed within 5-7 business days
- Only original purchase price is refunded (shipping not included)
- Digital products are non-refundable after download

To request a refund, contact support@example.com with your order number."""
    },
    {
        "title": "How to update billing information",
        "content": """To update your billing information:
1. Log in to your account
2. Go to Settings → Billing
3. Click "Edit payment method"
4. Enter new card details
5. Click Save

Your next invoice will use the new payment method.
Past invoices are not affected."""
    },
    {
        "title": "Account deletion",
        "content": """To delete your account:
1. Log in to your account
2. Go to Settings → Account
3. Click "Delete account" at the bottom
4. Confirm deletion

Warning: Account deletion is permanent. All your data will be lost.
This cannot be undone.

Please contact support if you need a backup first."""
    },
]
