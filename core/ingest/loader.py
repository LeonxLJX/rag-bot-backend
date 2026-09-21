"""
Document Ingestion Pipeline — Load → Split → Embed
"""
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from config.settings import settings


class DocumentIngestor:
    """
    Ingest documents into the RAG system.
    
    Supports: PDF, TXT, MD
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[str]:
        """Load a single file and return text content."""
        ext = self._get_extension(file_path)

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return [d.page_content for d in docs]
        elif ext in (".txt", ".md", ".markdown"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            return [d.page_content for d in docs]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def split_text(self, texts: List[str]) -> List[str]:
        """Split text into chunks."""
        chunks = []
        for text in texts:
            chunks.extend(self.splitter.split_text(text))
        return chunks

    def ingest(self, file_paths: List[str]) -> List[str]:
        """Full pipeline: load + split."""
        all_chunks = []
        for path in file_paths:
            try:
                texts = self.load_file(path)
                chunks = self.split_text(texts)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
        return all_chunks

    def _get_extension(self, path: str) -> str:
        return "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
