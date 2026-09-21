"""
Agent Tools Factory — Create tools for the LLM to use.

Inspired by Langchain-Chatchat's agent/tools_factory module.
Provides tools like:
- Search knowledge base
- Web search (stub)
- Calculator
- Database query (stub)
"""
from typing import List, Dict, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class KnowledgeBaseSearchInput(BaseModel):
    """Input for knowledge base search."""
    query: str = Field(description="The search query")
    top_k: int = Field(default=5, description="Number of results to return")


class CalculatorInput(BaseModel):
    """Input for calculator."""
    expression: str = Field(description="Mathematical expression to evaluate")


def create_kb_search_tool(retriever):
    """Create a tool for searching the knowledge base."""

    @tool("search_knowledge_base", args_schema=KnowledgeBaseSearchInput)
    def search_kb(query: str, top_k: int = 5) -> str:
        """
        Search the internal knowledge base for relevant information.
        Use this when the user asks about company documents, policies, or internal knowledge.
        """
        docs = retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs[:top_k]):
            results.append(f"[Source {i+1}]\n{doc.page_content[:500]}...")
        return "\n\n".join(results)

    return search_kb


def create_calculator_tool():
    """Create a calculator tool."""

    @tool("calculator", args_schema=CalculatorInput)
    def calculator(expression: str) -> str:
        """
        Evaluate a mathematical expression.
        Use this for any calculations, math, or numerical operations.
        """
        try:
            # Safe evaluation
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    return calculator


def get_default_tools(retriever=None) -> List:
    """Get default set of tools."""
    tools = []

    if retriever:
        tools.append(create_kb_search_tool(retriever))

    tools.append(create_calculator_tool())

    return tools
