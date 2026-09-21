"""
RAG Evaluation — Measure RAG quality quantitatively.

Industry-standard metrics:
- Context Precision: Is the right document in the retrieved set?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Is the answer actually relevant to the question?

This is how real teams know if their RAG is good or not.
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings


@dataclass
class EvalResult:
    """Result from RAG evaluation."""
    question: str
    answer: str
    context: str
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    overall_score: float


class RAGEvaluator:
    """
    Evaluate RAG system quality.
    
    Uses LLM-as-judge approach to measure:
    1. Context Precision: Did we retrieve the right docs?
    2. Faithfulness: Is the answer grounded?
    3. Answer Relevancy: Does the answer actually answer the question?
    
    This is how you know if your RAG tuning is working.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0,
        )

    def _score_context_precision(self, question: str, context: str) -> float:
        """Score: is the retrieved context actually relevant?"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate how relevant the given context is to answer the question.

Give a score from 0 to 10.
0 = completely irrelevant
10 = perfectly relevant

Context: {context}
Question: {question}

Score (just the number):"""),
        ])
        chain = prompt | self.llm
        result = chain.invoke({"context": context, "question": question})
        try:
            score = float(result.content.strip())
            return min(10.0, max(0.0, score)) / 10.0
        except:
            return 0.5

    def _score_faithfulness(self, context: str, answer: str) -> float:
        """Score: is the answer grounded in the context?"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate how well the answer is supported by the context.

Give a score from 0 to 10.
0 = completely hallucinated
10 = perfectly grounded in context

Context: {context}
Answer: {answer}

Score (just the number):"""),
        ])
        chain = prompt | self.llm
        result = chain.invoke({"context": context, "answer": answer})
        try:
            score = float(result.content.strip())
            return min(10.0, max(0.0, score)) / 10.0
        except:
            return 0.5

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        """Score: does the answer actually answer the question?"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate how well the answer answers the question.

Give a score from 0 to 10.
0 = completely irrelevant
10 = perfectly answers the question

Question: {question}
Answer: {answer}

Score (just the number):"""),
        ])
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "answer": answer})
        try:
            score = float(result.content.strip())
            return min(10.0, max(0.0, score)) / 10.0
        except:
            return 0.5

    def evaluate_single(
        self, question: str, answer: str, context: str
    ) -> EvalResult:
        """Evaluate a single RAG interaction."""
        cp = self._score_context_precision(question, context)
        faith = self._score_faithfulness(context, answer)
        relev = self._score_answer_relevancy(question, answer)
        overall = (cp + faith + relev) / 3

        return EvalResult(
            question=question,
            answer=answer,
            context=context,
            context_precision=cp,
            faithfulness=faith,
            answer_relevancy=relev,
            overall_score=overall,
        )

    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate a batch of test cases.
        
        test_cases format:
        [
            {"question": "...", "answer": "...", "context": "..."},
            ...
        ]
        """
        results = []
        for tc in test_cases:
            r = self.evaluate_single(tc["question"], tc["answer"], tc["context"])
            results.append(r)

        avg_cp = sum(r.context_precision for r in results) / len(results)
        avg_faith = sum(r.faithfulness for r in results) / len(results)
        avg_relev = sum(r.answer_relevancy for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)

        return {
            "num_test_cases": len(results),
            "avg_context_precision": round(avg_cp, 3),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_answer_relevancy": round(avg_relev, 3),
            "avg_overall_score": round(avg_overall, 3),
            "individual_results": [
                {
                    "question": r.question,
                    "overall_score": r.overall_score,
                }
                for r in results
            ],
        }


# ─── Example Eval Test Set ─────────────────────────────────────────────────────

EVAL_TEST_SET = [
    {
        "question": "What is RAG?",
        "expected_keywords": ["retrieval", "augmented", "generation"],
    },
    {
        "question": "How does hybrid search work?",
        "expected_keywords": ["vector", "BM25", "ensemble"],
    },
    {
        "question": "Why use reranking?",
        "expected_keywords": ["precision", "relevance", "rerank"],
    },
    {
        "question": "What is a knowledge base?",
        "expected_keywords": ["documents", "chunks", "vector"],
    },
]
