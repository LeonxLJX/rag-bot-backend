"""
Async Tasks — Background task queue for document processing.

Uses FastAPI BackgroundTasks for simple async processing.
In production, use Celery + Redis for distributed task queue.
"""
from fastapi import BackgroundTasks
from typing import List
import asyncio

from core.ingest.loader import DocumentIngestor
from core.rag.engine import RAGEngine
from config.settings import settings


# Global task status tracking
task_status: dict[str, dict] = {}


class TaskManager:
    """
    Manage background tasks.
    
    Tracks:
    - Task ID
    - Status (pending / processing / completed / failed)
    - Progress
    - Result / Error
    """

    def __init__(self):
        self.tasks: dict[str, dict] = {}

    def create_task(self, task_id: str, task_type: str):
        """Create a new task."""
        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
        }

    def update_progress(self, task_id: str, progress: int, status: str = "processing"):
        """Update task progress."""
        if task_id in self.tasks:
            self.tasks[task_id]["progress"] = progress
            self.tasks[task_id]["status"] = status

    def complete_task(self, task_id: str, result: dict):
        """Mark task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["progress"] = 100
            self.tasks[task_id]["result"] = result

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = error

    def get_task(self, task_id: str) -> dict:
        """Get task status."""
        return self.tasks.get(task_id, {})


# Global task manager
task_manager = TaskManager()


async def process_document_task(task_id: str, file_paths: List[str], kb_id: str):
    """
    Background task: process documents and build vector store.
    
    Steps:
    1. Load documents
    2. Split into chunks
    3. Build vector store
    4. Build retriever
    5. Build chain
    """
    try:
        task_manager.update_progress(task_id, 10, "processing")

        # Step 1: Load + split
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest(file_paths)
        task_manager.update_progress(task_id, 40, "processing")

        # Step 2: Build RAG
        engine = RAGEngine(kb_id=kb_id)
        engine.build(chunks)
        task_manager.update_progress(task_id, 80, "processing")

        # Step 3: Done
        task_manager.complete_task(task_id, {
            "chunks_processed": len(chunks),
            "kb_id": kb_id,
        })

    except Exception as e:
        task_manager.fail_task(task_id, str(e))
