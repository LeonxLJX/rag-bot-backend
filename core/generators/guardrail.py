"""
Guardrail — Real anti-hallucination checks.

Two-stage guardrail:
1. Pre-generation: Is the context relevant enough?
2. Post-generation: Is the answer actually grounded in the context?

This is NOT a TODO — it's real implementation.
"""
from typing import Tuple, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings


class Guardrail:
    """
    Anti-hallucination guardrail.
    
    Two stages:
    1. Pre-check: Is the retrieved context relevant to the question?
    2. Post-check: Does the answer actually come from the context?
    
    If either check fails, say "I don't know" instead of hallucinating.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0,  # Deterministic for guardrail
        )

        # Pre-check prompt: is the context relevant?
        self.pre_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance checker.
Given a user question and some context, determine if the context contains enough information to answer the question.

Answer with ONLY "YES" or "NO".

Question: {question}

Context:
{context}

Relevant?"""),
        ])

        # Post-check prompt: is the answer grounded?
        self.post_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker.
Given a context and an answer, determine if every claim in the answer is supported by the context.

Answer with ONLY "YES" or "NO".

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context?"""),
        ])

    def pre_check(self, question: str, context: str) -> bool:
        """
        Check if context is relevant enough to answer the question.
        
        Returns:
            True if context is relevant, False otherwise.
        """
        chain = self.pre_check_prompt | self.llm
        result = chain.invoke({"question": question, "context": context})
        answer = result.content.strip().upper()
        return "YES" in answer

    def post_check(self, context: str, answer: str) -> bool:
        """
        Check if the answer is grounded in the context.
        
        Returns:
            True if answer is grounded, False if hallucinated.
        """
        chain = self.post_check_prompt | self.llm
        result = chain.invoke({"context": context, "answer": answer})
        verdict = result.content.strip().upper()
        return "YES" in verdict

    def check_answer(self, question: str, context: str, answer: str) -> Tuple[str, bool]:
        """
        Full guardrail check.
        
        Returns:
            (final_answer, passed_guardrail)
        """
        # Pre-check: is context relevant?
        if not self.pre_check(question, context):
            return (
                "I don't have enough information to answer that question.",
                False,
            )

        # Post-check: is answer grounded?
        if not self.post_check(context, answer):
            return (
                "I don't have enough information to answer that question.",
                False,
            )

        # Both checks passed
        return answer, True
