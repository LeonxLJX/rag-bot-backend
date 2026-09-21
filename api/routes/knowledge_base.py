"""
Knowledge Base Routes — Document management endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import os

from core.auth.jwt_auth import get_current_user
from core.ingest.loader import DocumentIngestor
from core.rag.engine import RAGEngine
from api.routes.chat import rag_engines
from config.settings import settings

router = APIRouter(prefix="/api/knowledge-base", tags=["knowledge-base"])

ingestor = DocumentIngestor()


class KBResponse(BaseModel):
    id: str
    name: str
    document_count: int
    chunk_count: int


@router.post("/{kb_id}/upload")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a document to a knowledge base (requires JWT).

    - **kb_id**: Knowledge base ID
    - **file**: PDF, TXT, or MD file
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in DocumentIngestor.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: PDF, TXT, MD",
        )

    # Save file
    upload_dir = f"data/uploads/{kb_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Ingest into RAG
    chunks = ingestor.ingest([file_path])

    # Get or create RAG engine
    if kb_id not in rag_engines:
        rag_engines[kb_id] = RAGEngine(kb_id=kb_id)

    engine = rag_engines[kb_id]

    # Rebuild vector store with new chunks
    all_chunks = engine.chunks + chunks
    engine.build(all_chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks": len(all_chunks),
    }


@router.get("/{kb_id}/stats")
def get_kb_stats(kb_id: str):
    """Get statistics for a knowledge base."""
    if kb_id not in rag_engines:
        return {"kb_id": kb_id, "chunk_count": 0, "has_chain": False}

    engine = rag_engines[kb_id]
    return {
        "kb_id": kb_id,
        "chunk_count": len(engine.chunks),
        "has_chain": engine.chain is not None,
        "has_retriever": engine.retriever is not None,
    }


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a knowledge base (requires JWT)."""
    if kb_id in rag_engines:
        del rag_engines[kb_id]
    return {"status": "deleted", "kb_id": kb_id}
