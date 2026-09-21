"""
Auth Routes — login endpoint wiring the JWT module into the API.

Users live in the SQLAlchemy `users` table (core.db.models.DBUser). The first
login against an empty database seeds a demo account so the project is runnable
out of the box; production should provision users via signup/SSO instead.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.jwt_auth import (
    Token, create_access_token, get_password_hash, verify_password,
)
from core.db.models import DBUser, get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo1234"  # seeded only when the users table is empty


@router.post("/login", response_model=Token)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Exchange username/password for a JWT access token."""
    if db.query(DBUser).count() == 0:
        db.add(DBUser(
            username=DEMO_USERNAME,
            email="demo@example.com",
            hashed_password=get_password_hash(DEMO_PASSWORD),
            full_name="Demo User",
        ))
        db.commit()

    user = db.query(DBUser).filter(DBUser.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    token = create_access_token({"sub": user.username, "user_id": user.id})
    return Token(access_token=token)
