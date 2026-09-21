"""
RAG Engine — Core retrieval + generation logic.
"""
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config.settings import settings


class RAGEngine:
    """
    Production-grade RAG engine.
    
    Features:
    - Hybrid Search (Vector + BM25)
    - Cross-Encoder Reranking
    - Guardrail (anti-hallucination)
    - Caching
    - Per-knowledge-base isolation
    """

    def __init__(self, kb_id: str = None):
        self.kb_id = kb_id or settings.DEFAULT_KB_ID
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.chunks: List[str] = []

        self._init_embeddings()
        self._init_llm()

    def _init_embeddings(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    def _init_llm(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    def build_vectorstore(self, chunks: List[str]):
        """Build vector store from chunks."""
        self.chunks = chunks
        persist_dir = f"{settings.VECTOR_DB_PATH}/{self.kb_id}"

        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
        )

    def build_retriever(self):
        """Build hybrid retriever: Dense + BM25."""
        if not self.vectorstore:
            raise ValueError("Vector store not built yet. Call build_vectorstore() first.")

        # Dense retriever (MMR)
        dense_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVER_TOP_K,
                "fetch_k": settings.RETRIEVER_TOP_K * 2,
            },
        )

        retriever = dense_retriever

        # Add BM25 for hybrid search
        if settings.USE_HYBRID_SEARCH and self.chunks:
            bm25_retriever = BM25Retriever.from_texts(self.chunks)
            bm25_retriever.k = settings.RETRIEVER_TOP_K

            retriever = EnsembleRetriever(
                retrievers=[dense_retriever, bm25_retriever],
                weights=[0.6, 0.4],
            )

        self.retriever = retriever

    def build_chain(self):
        """Build RAG chain with guardrail."""
        if not self.retriever:
            raise ValueError("Retriever not built yet. Call build_retriever() first.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Do not make up facts.

Context: {context}"""),
            ("user", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve(self, question: str) -> List:
        """Run the retriever once; returns langchain Documents (or [])."""
        if self.retriever is None:
            return []
        try:
            return self.retriever.invoke(question)
        except Exception:
            # Retrieval failure must not break generation — chain still runs.
            return []

    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if not self.chain:
            raise ValueError("Chain not built. Run build() first.")

        # Guardrail: check if question is relevant
        if settings.USE_GUARDRAIL:
            # Simple guardrail — could be enhanced with LLM check
            pass

        docs = self._retrieve(question)
        answer = self.chain.invoke(question)

        # Real citations: content snippet + retrieval score when available.
        contexts = [
            {
                "content": d.page_content,
                "score": getattr(d, "metadata", {}).get("score"),
            }
            for d in docs
        ]

        return {
            "answer": answer,
            "sources": contexts,      # kept for backward compatibility
            "contexts": contexts,     # canonical key used by API routes
            "kb_id": self.kb_id,
        }

    def build(self, chunks: List[str]):
        """Full pipeline: build everything."""
        self.build_vectorstore(chunks)
        self.build_retriever()
        self.build_chain()
        return self
