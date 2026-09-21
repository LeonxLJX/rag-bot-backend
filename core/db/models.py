"""
Database Models — SQLAlchemy ORM models.

Models:
- User
- KnowledgeBase
- Document
- ChatSession
- ChatMessage
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from config.settings import settings


# ─── Database Setup ───────────────────────────────────────────────────────────

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ─────────────────────────────────────────────────────────────────────

class DBUser(Base):
    """User table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    knowledge_bases = relationship("DBKnowledgeBase", back_populates="owner")
    chat_sessions = relationship("DBChatSession", back_populates="user")


class DBKnowledgeBase(Base):
    """Knowledge Base table."""
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("DBUser", back_populates="knowledge_bases")
    documents = relationship("DBDocument", back_populates="kb")


class DBDocument(Base):
    """Document table."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    kb = relationship("DBKnowledgeBase", back_populates="documents")


class DBChatSession(Base):
    """Chat Session table."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    kb_id = Column(String)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("DBUser", back_populates="chat_sessions")
    messages = relationship("DBChatMessage", back_populates="session")


class DBChatMessage(Base):
    """Chat Message table."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)  # user / assistant
    content = Column(Text)
    sources = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("DBChatSession", back_populates="messages")


# ─── Init ───────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
