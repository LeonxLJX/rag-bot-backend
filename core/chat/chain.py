"""
Chat Chain — Conversation-aware RAG chain.

Handles:
- Multi-turn conversation memory
- Query rewriting (standalone question generation)
- Guardrail (anti-hallucination)
- Streaming output

Inspired by Langchain-Chatchat's chat module.
"""
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import settings


class ChatHistory:
    """In-memory chat history."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict]:
        return self.messages

    def clear(self):
        self.messages = []


class ConversationRAGChain:
    """
    Conversation-aware RAG chain with memory.
    
    Features:
    - Multi-turn conversation support
    - Query rewriting (standalone question)
    - Context window management
    - Guardrail
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )
        self.histories: Dict[str, ChatHistory] = {}

        # Query rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question, 
reformulate the question to be a standalone question that can be understood 
without the chat history.

Chat history: {chat_history}
User question: {question}

Standalone question:"""),
        ])

        # Answer prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
Answer based ONLY on the provided context.
If you don't know, say "I don't have enough information."

Context: {context}
Question: {question}"""),
        ])

    def get_or_create_history(self, session_id: str) -> ChatHistory:
        if session_id not in self.histories:
            self.histories[session_id] = ChatHistory(session_id)
        return self.histories[session_id]

    def rewrite_query(self, question: str, chat_history: List[Dict]) -> str:
        """Rewrite follow-up question to standalone question."""
        if not chat_history:
            return question

        history_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in chat_history[-6:]  # last 3 turns
        ])

        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({
            "chat_history": history_text,
            "question": question,
        })

        return rewritten.strip()

    def format_docs(self, docs) -> str:
        """Format retrieved documents into context string."""
        return "\n\n".join([d.page_content for d in docs])

    def chat(self, question: str, session_id: str) -> Dict:
        """
        Main chat function with conversation memory.
        
        Pipeline:
        1. Get chat history
        2. Rewrite question (standalone)
        3. Retrieve relevant docs
        4. Generate answer
        5. Update history
        """
        history = self.get_or_create_history(session_id)

        # Step 1: Rewrite question
        standalone_q = self.rewrite_query(question, history.get_messages())

        # Step 2: Retrieve
        docs = self.retriever.invoke(standalone_q)
        context = self.format_docs(docs)

        # Step 3: Generate
        answer_chain = self.answer_prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "question": standalone_q,
        })

        # Step 4: Update history
        history.add_message("user", question)
        history.add_message("assistant", answer)

        return {
            "answer": answer,
            "standalone_question": standalone_q,
            "num_sources": len(docs),
            "session_id": session_id,
        }
