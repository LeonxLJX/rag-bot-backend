"""
Vector Store Factory — Choose between Chroma (dev) and Pinecone (prod).

Production-ready switch:
- Dev: Chroma (local, no setup needed)
- Prod: Pinecone (managed, scalable)

Switch via VECTOR_STORE_TYPE env var.
"""
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


def get_vectorstore(embeddings: OpenAIEmbeddings, texts: List[str] = None):
    """
    Get vector store instance based on config.
    
    Dev: Chroma (local)
    Prod: Pinecone (managed)
    """
    store_type = settings.VECTOR_STORE_TYPE

    if store_type == "pinecone":
        return _create_pinecone_store(embeddings, texts)
    else:
        return _create_chroma_store(embeddings, texts)


def _create_chroma_store(embeddings: OpenAIEmbeddings, texts: List[str] = None):
    """Create Chroma vector store (development)."""
    persist_dir = f"{settings.VECTOR_DB_PATH}/{settings.DEFAULT_KB_ID}"

    if texts:
        return Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    else:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )


def _create_pinecone_store(embeddings: OpenAIEmbeddings, texts: List[str] = None):
    """
    Create Pinecone vector store (production).
    
    Requires:
    - PINECONE_API_KEY
    - PINECONE_ENVIRONMENT
    - PINECONE_INDEX_NAME
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        import pinecone

        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT,
        )

        if texts:
            return PineconeVectorStore.from_texts(
                texts=texts,
                embedding=embeddings,
                index_name=settings.PINECONE_INDEX_NAME,
            )
        else:
            return PineconeVectorStore(
                embedding=embeddings,
                index_name=settings.PINECONE_INDEX_NAME,
            )

    except ImportError:
        raise ImportError(
            "pinecone-client not installed. "
            "Install with: pip install pinecone-client"
        )
