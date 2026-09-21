"""
Reranker — Cross-encoder reranking for retrieved documents.

Improves retrieval quality by re-scoring candidates with a more powerful model.
Inspired by Langchain-Chatchat's reranker module.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoredDoc:
    """Document with relevance score."""
    content: str
    score: float
    metadata: dict = None


class BaseReranker:
    """Base class for rerankers."""

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[ScoredDoc]:
        raise NotImplementedError


class SimpleReranker(BaseReranker):
    """
    Simple reranker based on keyword overlap.
    
    For production, use CrossEncoderReranker with a real model.
    This is a lightweight fallback.
    """

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[ScoredDoc]:
        query_words = set(query.lower().split())
        scored = []

        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scored.append(ScoredDoc(content=doc, score=score))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:top_n]


class CrossEncoderReranker(BaseReranker):
    """
    Production-grade reranker using Cross-Encoder model.
    
    Uses BAAI/bge-reranker or similar cross-encoder model
    to re-score query-document pairs.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[ScoredDoc]:
        """Rerank documents using cross-encoder."""
        self._load_model()

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores
        scores = self._model.predict(pairs)

        # Combine and sort
        scored = [
            ScoredDoc(content=doc, score=float(score))
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:top_n]


def get_reranker(reranker_type: str = "simple") -> BaseReranker:
    """Factory function to get reranker instance."""
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    return SimpleReranker()
