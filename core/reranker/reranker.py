"""
Reranker — Production-grade Cross-Encoder reranking.

Real implementation using sentence-transformers CrossEncoder.
Not keyword overlap — actual neural reranking.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os


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


class CrossEncoderReranker(BaseReranker):
    """
    Production-grade reranker using Cross-Encoder model.
    
    Uses BAAI/bge-reranker-base or similar cross-encoder model
    to re-score query-document pairs.
    
    This is REAL reranking, not keyword overlap.
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
                print(f"✅ Loaded reranker: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[ScoredDoc]:
        """
        Rerank documents using cross-encoder.
        
        Instead of just counting keyword overlap,
        the model reads each (query, document) pair and
        predicts relevance.
        
        This is how real production RAG works.
        """
        self._load_model()

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs)

        # Combine and sort by score
        scored = [
            ScoredDoc(content=doc, score=float(score))
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:top_n]


class SimpleReranker(BaseReranker):
    """
    Lightweight fallback reranker.
    Used when sentence-transformers is not available.
    """

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[ScoredDoc]:
        """Simple TF-IDF based reranking."""
        query_words = set(query.lower().split())
        scored = []

        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scored.append(ScoredDoc(content=doc, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]


def get_reranker(reranker_type: str = "cross_encoder") -> BaseReranker:
    """
    Factory function to get reranker instance.
    
    Args:
        reranker_type: "cross_encoder" (production) or "simple" (dev)
    """
    if reranker_type == "cross_encoder":
        try:
            return CrossEncoderReranker()
        except ImportError:
            print("⚠️ sentence-transformers not available, falling back to SimpleReranker")
            return SimpleReranker()
    return SimpleReranker()
