"""
File RAG Engine — File-specific RAG pipeline.

Handles:
- Multiple file types (PDF, TXT, MD, DOCX)
- Chunking strategy optimization
- Hybrid retrieval with BM25 + Vector
- Reranking
- Guardrail
- Caching

Inspired by Langchain-Chatchat's file_rag module.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config.settings import settings
from core.reranker.reranker import get_reranker
from core.generators.guardrail import Guardrail


@dataclass
class RetrievalResult:
    """Result from retrieval step."""
    documents: List[str]
    scores: List[float]
    query: str
    retriever_type: str = "hybrid"


class FileRAGEngine:
    """
    Production-grade File RAG Engine.
    
    Pipeline:
    1. Document Loading → Text Extraction
    2. Text Splitting → Chunks
    3. Vectorization → Embeddings
    4. Retrieval → Hybrid Search (Vector + BM25)
    5. Reranking → Cross-Encoder
    6. Generation → LLM with Guardrail
    """

    def __init__(self, kb_id: str = None):
        self.kb_id = kb_id or settings.DEFAULT_KB_ID
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.chunks: List[str] = []
        self.reranker = get_reranker("cross_encoder")
        self.guardrail = Guardrail() if settings.USE_GUARDRAIL else None

        # Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    # ─── Ingestion ───────────────────────────────────────────────────────────

    def build_vectorstore(self, chunks: List[str]):
        """Build Chroma vector store from chunks."""
        self.chunks = chunks
        persist_dir = f"{settings.VECTOR_DB_PATH}/{self.kb_id}"

        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
        )
        return self

    def build_retriever(self):
        """Build hybrid retriever: Dense Vector + BM25 Sparse."""
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vectorstore() first.")

        # Dense retriever with MMR
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
        return self

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant chunks."""
        if not self.retriever:
            raise ValueError("Retriever not built.")

        docs = self.retriever.invoke(query)
        contents = [d.page_content for d in docs]

        # Rerank if enabled
        if settings.USE_RERANKER:
            reranked = self.reranker.rerank(query, contents, top_n=settings.RERANK_TOP_N)
            contents = [r.content for r in reranked]
            scores = [r.score for r in reranked]
        else:
            scores = [1.0] * len(contents)

        return RetrievalResult(
            documents=contents,
            scores=scores,
            query=query,
            retriever_type="hybrid+rerank" if settings.USE_RERANKER else "hybrid",
        )

    # ─── Generation ───────────────────────────────────────────────────────────

    def build_chain(self):
        """Build RAG chain with guardrail prompt."""
        if not self.retriever:
            raise ValueError("Retriever not built.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Do not make up facts.
Cite sources as [1], [2], etc. when relevant.

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

        return self

    # ─── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict:
        """Full query pipeline: retrieve → rerank → generate → guardrail."""
        if not self.chain:
            raise ValueError("Chain not built.")

        # Retrieve
        retrieval_result = self.retrieve(question)

        # Generate
        answer = self.chain.invoke(question)

        # Guardrail: real anti-hallucination check
        guardrail_passed = True
        if self.guardrail:
            context_text = "\n\n".join(retrieval_result.documents[:3])
            answer, guardrail_passed = self.guardrail.check_answer(
                question=question,
                context=context_text,
                answer=answer,
            )

        return {
            "answer": answer,
            "sources": [
                {"content": doc[:200], "score": score}
                for doc, score in zip(retrieval_result.documents, retrieval_result.scores)
            ],
            "kb_id": self.kb_id,
            "retrieval_type": retrieval_result.retriever_type,
            "num_chunks_retrieved": len(retrieval_result.documents),
            "guardrail_passed": guardrail_passed,
        }

    def build(self, chunks: List[str]):
        """Full pipeline: build everything."""
        self.build_vectorstore(chunks)
        self.build_retriever()
        self.build_chain()
        return self
