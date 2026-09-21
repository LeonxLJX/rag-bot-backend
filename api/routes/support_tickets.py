"""
Ticket System RAG — Support ticket chatbot.

Scenario: A company gets 500 support tickets/week.
Goal: Auto-resolve 40% of repetitive tickets.

API Endpoints:
- POST /api/tickets/create — Create a ticket
- POST /api/tickets/{id}/reply — Reply to ticket
- GET /api/tickets — List tickets
- GET /api/tickets/{id} — Get ticket details
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

from core.auth.jwt_auth import get_current_user
from core.logging.structured_logger import logger


router = APIRouter(prefix="/api/tickets", tags=["support-tickets"])


# ─── Data Models ──────────────────────────────────────────────────────────────

class TicketCreate(BaseModel):
    subject: str
    description: str
    priority: str = "medium"  # low, medium, high, urgent
    category: str = "general"  # billing, technical, account, other


class TicketReply(BaseModel):
    message: str


class Ticket(BaseModel):
    id: str
    subject: str
    description: str
    priority: str
    category: str
    status: str  # open, pending, resolved, closed
    created_at: str
    replies: List[dict] = []


# ─── In-Memory Ticket Store (demo) ─────────────────────────────────────────────

tickets: dict[str, dict] = {}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/create")
async def create_ticket(
    request: TicketCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a new support ticket."""
    ticket_id = str(uuid.uuid4())

    ticket = {
        "id": ticket_id,
        "subject": request.subject,
        "description": request.description,
        "priority": request.priority,
        "category": request.category,
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
        "created_by": current_user.get("username"),
        "replies": [],
        "auto_resolved": False,
    }

    # Try to auto-resolve using RAG
    # In production: query the support KB
    # If confidence > 0.7, auto-reply and mark as "auto-resolved"

    tickets[ticket_id] = ticket

    logger.info(
        "Ticket created",
        extra={
            "ticket_id": ticket_id,
            "subject": request.subject,
            "priority": request.priority,
        },
    )

    return ticket


@router.get("")
async def list_tickets(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """List all tickets."""
    user_tickets = [
        t for t in tickets.values()
        if t["created_by"] == current_user.get("username")
    ]

    if status:
        user_tickets = [t for t in user_tickets if t["status"] == status]

    return {
        "tickets": user_tickets,
        "total": len(user_tickets),
    }


@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get ticket details."""
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return tickets[ticket_id]


@router.post("/{ticket_id}/reply")
async def reply_ticket(
    ticket_id: str,
    request: TicketReply,
    current_user: dict = Depends(get_current_user),
):
    """Reply to a ticket."""
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket = tickets[ticket_id]
    ticket["replies"].append({
        "author": current_user.get("username"),
        "message": request.message,
        "timestamp": datetime.utcnow().isoformat(),
    })

    logger.info(
        "Ticket reply",
        extra={"ticket_id": ticket_id, "author": current_user.get("username")},
    )

    return ticket


@router.post("/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Resolve a ticket."""
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail="Ticket not found")

    tickets[ticket_id]["status"] = "resolved"

    return {
        "ticket_id": ticket_id,
        "status": "resolved",
    }
