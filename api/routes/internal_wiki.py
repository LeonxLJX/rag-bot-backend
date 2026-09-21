"""
Internal Wiki RAG — Company internal knowledge base.

Scenario: A company has HR policies, IT docs, onboarding materials.
Goal: New employees can self-serve instead of pinging HR/IT.

API Endpoints:
- POST /api/wiki/chat — Employee asks a question
- GET /api/wiki/departments — List departments (HR, IT, Finance)
- GET /api/wiki/policies/{dept} — Get all policies for a department
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from core.auth.jwt_auth import get_current_user
from core.logging.structured_logger import logger


router = APIRouter(prefix="/api/wiki", tags=["internal-wiki"])


# ─── Data Models ──────────────────────────────────────────────────────────────

class WikiChatRequest(BaseModel):
    question: str
    department: Optional[str] = None  # HR, IT, Finance, Engineering
    session_id: Optional[str] = None


# ─── Departments ──────────────────────────────────────────────────────────────

DEPARTMENTS = {
    "HR": {
        "name": "Human Resources",
        "policies": [
            {
                "title": "PTO Policy",
                "content": """Paid Time Off (PTO):
- Full-time employees accrue 15 days per year
- PTO can be used for vacation, sick days, or personal days
- Unused PTO rolls over up to 5 days per year
- Request PTO at least 2 weeks in advance for periods >3 days
- Manager approval required for all PTO requests

PTO accrual: 1.25 days per month
Payout: Not paid out upon termination"""
            },
            {
                "title": "Remote Work Policy",
                "content": """Remote Work Policy:
- Hybrid: 3 days in office, 2 days remote
- Fully remote: Available for certain roles (Engineering, Design)
- Home office stipend: $500/year for equipment
- Must be online during core hours (10am-4pm local time)
- Performance reviews based on output, not hours worked

Security: Use company VPN. Do not share login credentials."""
            },
            {
                "title": "Health Insurance",
                "content": """Health Insurance:
- Company covers 80% of premium for employee
- Family coverage: $200/month additional
- Plans: PPO (network providers) or HSA (high deductible)
- HSA: Company contributes $1000/year
- Coverage starts day 1
- Dental and vision included

Questions: contact hr@example.com"""
            },
        ],
    },
    "IT": {
        "name": "IT Department",
        "policies": [
            {
                "title": "VPN Setup",
                "content": """How to set up VPN:
1. Download Cisco AnyConnect from it.example.com
2. Install and open the app
3. Enter VPN address: vpn.example.com
4. Use your SSO credentials
5. Duo two-factor authentication required

Troubleshooting:
- Can't connect? Check your internet connection
- Duo not working? Call IT at ext. 4567
- VPN disconnects? Reconnect, it should auto-reconnect in 30 seconds"""
            },
            {
                "title": "Password Requirements",
                "content": """Password Requirements:
- Minimum 12 characters
- Must include: uppercase, lowercase, number, special character
- Cannot reuse last 5 passwords
- Must change every 90 days
- Cannot write down passwords
- Use password manager (1Password)

If your account is locked: Call IT at ext. 4567"""
            },
            {
                "title": "Laptop Setup",
                "content": """New Laptop Setup:
1. Unbox and connect to power
2. Turn on and follow setup wizard
3. Log in with your SSO
4. Install required software:
   - 1Password
   - Slack
   - VS Code
   - Chrome
5. Request software access through IT portal

IT contact: it@example.com or ext. 4567"""
            },
        ],
    },
    "Finance": {
        "name": "Finance Department",
        "policies": [
            {
                "title": "Expense Report",
                "content": """Expense Report Policy:
- Submit expenses within 30 days of purchase
- All expenses require receipts
- Meals: $25 per person per meal
- Hotels: Up to $200/night
- Flights: Economy class only
- Uber/Lyft: Up to $30 per ride

Submit through: Expensify app
Approval: Direct manager → Finance
Reimbursement: 5-7 business days after approval"""
            },
            {
                "title": "401(k) Plan",
                "content": """401(k) Retirement Plan:
- Company match: 50% of first 6% contributed
- Vesting: 3-year cliff (0% before year 3, 100% after)
- Contribution limit: $23,000/year (2024)
- After-tax: Yes, up to $69,000 total
- Enrollment: Eligible after 90 days

Questions: Contact benefits@example.com"""
            },
        ],
    },
    "Engineering": {
        "name": "Engineering Department",
        "policies": [
            {
                "title": "Code Review Policy",
                "content": """Code Review Policy:
- All PRs require at least 1 approval
- Senior engineers approve for release branches
- PR should be reviewed within 24 hours
- PR should be under 400 lines of code
- Use conventional commits: feat/fix/chore/docs
- CI must pass before merge

Merge strategy: Squash and merge"""
            },
            {
                "title": "On-Call Rotation",
                "content": """On-Call Rotation:
- Rotation: 1 week per engineer
- On-call responsibilities:
  - Respond to alerts within 15 minutes
  - Page senior if can't fix in 30 minutes
  - Document all incidents
- Compensation: On-call stipend $200/week
- Time off in lieu: 8 hours after each on-call week

Escalation: Page SRE if severity > 2"""
            },
        ],
    },
}


@router.get("/departments")
async def list_departments():
    """List all departments with policies."""
    return {
        "departments": [
            {
                "id": dept_id,
                "name": info["name"],
                "num_policies": len(info["policies"]),
            }
            for dept_id, info in DEPARTMENTS.items()
        ]
    }


@router.get("/policies/{dept_id}")
async def list_policies(dept_id: str):
    """List all policies for a department."""
    if dept_id not in DEPARTMENTS:
        return {"error": "Department not found"}

    return {
        "department": dept_id,
        "policies": [
            {"title": p["title"], "content": p["content"][:200] + "..."}
            for p in DEPARTMENTS[dept_id]["policies"]
        ],
    }


@router.post("/chat")
async def wiki_chat(
    request: WikiChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Internal wiki chatbot.

    Employees can ask questions about company policies.
    Access control: JWT required (see core.auth.jwt_auth.get_current_user).

    Retrieval is a real lexical match over the seeded policy corpus for the
    requested department — no LLM call here, so the scenario works offline.
    A production deployment would swap this for the same RAGEngine pipeline
    used by /api/chat (embedding + hybrid search + generation), with each
    department backed by its own knowledge base id.
    """
    logger.info(
        "Wiki chat",
        extra={
            "user": current_user.get("username"),
            "question": request.question[:50],
            "department": request.department,
        },
    )

    dept_id = request.department if request.department in DEPARTMENTS else None
    # Search the requested department only; fall back to all when unset.
    scope = {dept_id: DEPARTMENTS[dept_id]} if dept_id else DEPARTMENTS

    question_terms = {w.strip(".,?!").lower() for w in request.question.split()} - {""}

    scored = []
    for d_id, dept in scope.items():
        for policy in dept["policies"]:
            haystack = (policy["title"] + " " + policy["content"]).lower()
            overlap = sum(1 for t in question_terms if t and t in haystack)
            if overlap:
                scored.append((overlap, d_id, policy))
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return {
            "answer": "I couldn't find a policy that matches your question. "
                      "Try rephrasing, or contact the relevant department directly.",
            "department": dept_id or "all",
            "sources": [],
            "confidence": 0.0,
        }

    top = scored[0]
    _, best_dept, best_policy = top
    max_overlap = top[0]
    confidence = round(min(1.0, max_overlap / max(len(question_terms), 1)), 2)

    return {
        "answer": f"{best_dept} — {best_policy['title']}:\n{best_policy['content']}",
        "department": best_dept,
        "sources": [f"{best_dept}/{best_policy['title']}"],
        "confidence": confidence,
    }
