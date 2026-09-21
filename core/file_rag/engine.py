"""
File RAG Engine — Production-grade RAG pipeline.

Pipeline:
1. Document Loading → Text Extraction
2. Text Splitting → Chunks
3. Vectorization → Embeddings
4. Retrieval → Hybrid Search (Vector + BM25)
5. Reranking → Cross-Encoder
6. Guardrail Pre-check → Is context relevant?
7. Generation → LLM with context
8. Guardrail Post-check → Is answer grounded?

No duplicate retrieval. Full control over every step.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    
    Every step is explicit and controllable:
    - Hybrid retrieval (vector + BM25)
    - Cross-Encoder reranking
    - Guardrail pre-check + post-check
    - Caching at query level
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
        self.dense_retriever = None
        self.bm25_retriever = None
        self.chunks: List[str] = []
        self.reranker = get_reranker("cross_encoder")
        self.guardrail = Guardrail() if settings.USE_GUARDRAIL else None

        # Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

        # Prompt template
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Do not make up facts.
Cite sources as [1], [2], etc. when relevant.

Context: {context}"""),
            ("user", "{question}"),
        ])

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

    def build_retrievers(self):
        """Build dense + sparse retrievers separately."""
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vectorstore() first.")

        # Dense retriever with MMR (maximal marginal relevance)
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVER_TOP_K,
                "fetch_k": settings.RETRIEVER_TOP_K * 2,
            },
        )

        # BM25 sparse retriever
        if settings.USE_HYBRID_SEARCH and self.chunks:
            self.bm25_retriever = BM25Retriever.from_texts(self.chunks)
            self.bm25_retriever.k = settings.RETRIEVER_TOP_K

        return self

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def _dense_retrieve(self, query: str) -> List[str]:
        """Dense vector retrieval."""
        docs = self.dense_retriever.invoke(query)
        return [d.page_content for d in docs]

    def _sparse_retrieve(self, query: str) -> List[str]:
        """BM25 sparse retrieval."""
        if not self.bm25_retriever:
            return []
        docs = self.bm25_retriever.invoke(query)
        return [d.page_content for d in docs]

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Hybrid retrieval: dense + sparse, deduplicated.
        Then rerank with Cross-Encoder.
        """
        # Dense results
        dense_docs = self._dense_retrieve(query)

        # Sparse results
        sparse_docs = self._sparse_retrieve(query)

        # Combine and deduplicate
        seen = set()
        combined = []
        for doc in dense_docs + sparse_docs:
            # Deduplicate by first 100 chars
            key = doc[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        # Rerank
        if settings.USE_RERANKER:
            reranked = self.reranker.rerank(
                query, combined, top_n=settings.RERANK_TOP_N
            )
            contents = [r.content for r in reranked]
            scores = [r.score for r in reranked]
            retriever_type = "hybrid+rerank"
        else:
            contents = combined[:settings.RERANK_TOP_N]
            scores = [1.0] * len(contents)
            retriever_type = "hybrid"

        return RetrievalResult(
            documents=contents,
            scores=scores,
            query=query,
            retriever_type=retriever_type,
        )

    # ─── Generation ───────────────────────────────────────────────────────────

    def _format_context(self, documents: List[str]) -> str:
        """Format retrieved documents into context string."""
        return "\n\n---\n\n".join(
            [f"[Source {i+1}]\n{doc}" for i, doc in enumerate(documents)]
        )

    def _generate(self, question: str, context: str) -> str:
        """Generate answer from LLM."""
        chain = self.answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    # ─── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict:
        """
        Full RAG pipeline:
        1. Retrieve (hybrid)
        2. Rerank
        3. Guardrail pre-check
        4. Generate
        5. Guardrail post-check
        """
        # Step 1-2: Retrieve + Rerank
        retrieval_result = self.retrieve(question)
        context = self._format_context(retrieval_result.documents)

        # Step 3: Guardrail pre-check
        if self.guardrail:
            pre_passed = self.guardrail.pre_check(question, context)
            if not pre_passed:
                return {
                    "answer": "I don't have enough information to answer that.",
                    "sources": [],
                    "kb_id": self.kb_id,
                    "retrieval_type": retrieval_result.retriever_type,
                    "num_chunks_retrieved": 0,
                    "guardrail_passed": False,
                    "guardrail_stage": "pre_check",
                }

        # Step 4: Generate
        answer = self._generate(question, context)

        # Step 5: Guardrail post-check
        guardrail_passed = True
        if self.guardrail:
            answer, guardrail_passed = self.guardrail.check_answer(
                question=question,
                context=context,
                answer=answer,
            )

        return {
            "answer": answer,
            "sources": [
                {"content": doc[:300], "score": round(score, 4)}
                for doc, score in zip(
                    retrieval_result.documents, retrieval_result.scores
                )
            ],
            "kb_id": self.kb_id,
            "retrieval_type": retrieval_result.retriever_type,
            "num_chunks_retrieved": len(retrieval_result.documents),
            "guardrail_passed": guardrail_passed,
        }

    def build(self, chunks: List[str]):
        """Full pipeline: build everything."""
        self.build_vectorstore(chunks)
        self.build_retrievers()
        return self
