"""
Product Docs RAG — Product documentation chatbot.

Scenario: A SaaS company has product docs, API docs, changelogs.
Goal: Customers and sales can self-serve instead of contacting support.

API Endpoints:
- POST /api/product/chat — Ask a product question
- GET /api/product/docs — List all docs
- GET /api/product/changelog — Recent changelog
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from core.auth.jwt_auth import get_current_user
from core.logging.structured_logger import logger


router = APIRouter(prefix="/api/product", tags=["product-docs"])


# ─── Data Models ──────────────────────────────────────────────────────────────

class ProductChatRequest(BaseModel):
    question: str
    product: Optional[str] = None  # API, Dashboard, Mobile App
    session_id: Optional[str] = None


# ─── Product Documentation ─────────────────────────────────────────────────────

PRODUCT_DOCS = {
    "API": [
        {
            "title": "Authentication",
            "content": """API Authentication:
- Use Bearer token in Authorization header
- Get API key from Dashboard → Settings → API Keys
- Tokens expire after 24 hours
- Rate limit: 100 requests/minute

Example:
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/v1/users

Errors:
- 401: Invalid token
- 429: Rate limit exceeded"""
        },
        {
            "title": "Pagination",
            "content": """Pagination:
- Use page and per_page query parameters
- Default: page=1, per_page=20
- Max per_page: 100
- Response includes: has_more, total, page, per_page

Example:
GET /v1/users?page=2&per_page=50"""
        },
        {
            "title": "Webhooks",
            "content": """Webhooks:
- Configure webhook URL in Dashboard
- Events: user.created, user.updated, payment.failed
- Signature: X-Webhook-Signature header
- Verify signature using your webhook secret

Retry policy: 3 retries with exponential backoff"""
        },
    ],
    "Dashboard": [
        {
            "title": "Getting Started",
            "content": """Dashboard Getting Started:
1. Log in at app.example.com
2. Create your first project
3. Invite team members
4. Set up billing

Projects are isolated workspaces. Each project has its own API keys."""
        },
        {
            "title": "Team Management",
            "content": """Team Management:
- Roles: Admin, Member, Viewer
- Admin: Full access + billing
- Member: Edit projects
- Viewer: Read-only
- Invite via email: Settings → Team → Invite
- Free plan: Up to 5 team members
- Pro plan: Unlimited"""
        },
    ],
    "Mobile App": [
        {
            "title": "Push Notifications",
            "content": """Push Notifications:
- Enable in Settings → Notifications
- Categories: Mentions, Assignments, Comments
- Quiet hours: 10pm-7am by default
- Customize per category
- iOS: Requires notification permission
- Android: Requires notification permission

Troubleshooting:
- Not getting notifications? Check system settings
- Android: Check battery optimization
- iOS: Check Focus mode"""
        },
        {
            "title": "Offline Mode",
            "content": """Offline Mode:
- Access recent data without internet
- Changes sync when back online
- Conflicts: Last write wins
- Cache: Last 7 days of data
- Storage: Up to 500MB

Limitations:
- Can't access new data offline
- Can't invite team members
- Can't change settings"""
        },
    ],
}


@router.get("/docs")
async def list_docs(product: Optional[str] = None):
    """List product documentation."""
    if product:
        if product not in PRODUCT_DOCS:
            return {"error": "Product not found"}
        return {
            "product": product,
            "docs": [
                {"title": d["title"], "content": d["content"][:200] + "..."}
                for d in PRODUCT_DOCS[product]
            ],
        }

    return {
        "products": [
            {
                "id": p,
                "num_docs": len(docs),
            }
            for p, docs in PRODUCT_DOCS.items()
        ]
    }


@router.get("/changelog")
async def changelog():
    """Recent changelog entries."""
    return {
        "changelog": [
            {
                "version": "2.1.0",
                "date": "2026-09-15",
                "changes": [
                    "Added dark mode",
                    "Improved search speed by 40%",
                    "Fixed bug: Webhook retries not working",
                ],
            },
            {
                "version": "2.0.0",
                "date": "2026-08-01",
                "changes": [
                    "New dashboard UI",
                    "API v2 released",
                    "Team management improvements",
                ],
            },
        ]
    }


@router.post("/chat")
async def product_chat(
    request: ProductChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Product docs chatbot.
    
    Customers and sales can ask product questions.
    """
    logger.info(
        "Product chat",
        extra={
            "user": current_user.get("username"),
            "question": request.question[:50],
            "product": request.product,
        },
    )

    return {
        "answer": f"Based on our product docs: {request.question}",
        "product": request.product or "general",
        "sources": ["docs/api/auth.md", "docs/api/pagination.md"],
        "confidence": 0.9,
    }
