"""
Knowledge Base Manager — Multi-tenant KB lifecycle management.

Inspired by Langchain-Chatchat's knowledge_base module.
Handles: create, list, delete, update knowledge bases.
"""
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from config.settings import settings


@dataclass
class KnowledgeBase:
    """Knowledge Base metadata."""
    kb_id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str = ""
    vector_store_type: str = "chroma"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.embedding_model:
            self.embedding_model = settings.EMBEDDING_MODEL


class KnowledgeBaseManager:
    """
    Manage multiple knowledge bases.
    
    Each KB has:
    - Its own vector store directory
    - Metadata stored in a JSON file
    - Isolated from other KBs (multi-tenant)
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or settings.VECTOR_DB_PATH
        self.meta_file = os.path.join(self.base_dir, "kb_meta.json")
        self._ensure_dir()
        self._load_meta()

    def _ensure_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_meta(self):
        """Load KB metadata from JSON file."""
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.kbs: Dict[str, KnowledgeBase] = {
                    k: KnowledgeBase(**v) for k, v in json.load(f).items()
                }
        else:
            self.kbs = {}

    def _save_meta(self):
        """Save KB metadata to JSON file."""
        data = {k: asdict(v) for k, v in self.kbs.items()}
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_kb(self, kb_id: str, name: str, description: str = "") -> KnowledgeBase:
        """Create a new knowledge base."""
        if kb_id in self.kbs:
            raise ValueError(f"Knowledge base {kb_id} already exists")

        kb = KnowledgeBase(
            kb_id=kb_id,
            name=name,
            description=description,
        )
        self.kbs[kb_id] = kb
        self._save_meta()

        # Create vector store directory
        os.makedirs(os.path.join(self.base_dir, kb_id), exist_ok=True)

        return kb

    def list_kbs(self) -> List[Dict]:
        """List all knowledge bases."""
        return [asdict(kb) for kb in self.kbs.values()]

    def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by ID."""
        return self.kbs.get(kb_id)

    def delete_kb(self, kb_id: str):
        """Delete a knowledge base and its vector store."""
        if kb_id not in self.kbs:
            raise ValueError(f"Knowledge base {kb_id} not found")

        del self.kbs[kb_id]
        self._save_meta()

        # TODO: delete vector store directory
        # import shutil
        # shutil.rmtree(os.path.join(self.base_dir, kb_id))

    def update_stats(self, kb_id: str, doc_count: int, chunk_count: int):
        """Update KB statistics."""
        if kb_id in self.kbs:
            self.kbs[kb_id].document_count = doc_count
            self.kbs[kb_id].chunk_count = chunk_count
            self.kbs[kb_id].updated_at = datetime.now().isoformat()
            self._save_meta()
