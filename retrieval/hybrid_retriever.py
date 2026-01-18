"""
Hybrid Retriever using LangChain's EnsembleRetriever
Combines semantic search (dense) with keyword search (sparse/BM25)
"""

import os
from typing import Optional
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def create_hybrid_retriever(
    video_id: Optional[str] = None,
    top_k: int = 4,
    namespace: str = "",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3
):
    """
    Create hybrid retriever using LangChain's EnsembleRetriever
    
    Combines:
    1. Dense retrieval (semantic/vector search) - LangChain Pinecone
    2. Sparse retrieval (keyword/BM25 search) - LangChain BM25Retriever
    
    From tutorial: This improves retrieval quality by 10-15%
    
    Args:
        video_id: Optional video filter
        top_k: Number of results
        namespace: Pinecone namespace
        dense_weight: Weight for semantic search (default: 0.7)
        sparse_weight: Weight for keyword search (default: 0.3)
    
    Returns:
        LangChain EnsembleRetriever
    
    Example:
        >>> retriever = create_hybrid_retriever(video_id="O5xeyoRL95U")
        >>> results = retriever.invoke("GPT-3 parameters")
        >>> # Gets semantic matches AND exact "GPT-3" keyword matches!
    """
    logger.info(
        f"Creating hybrid retriever: video_id={video_id}, "
        f"weights=({dense_weight}/{sparse_weight})"
    )
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    # Create semantic retriever (dense)
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    
    search_kwargs = {"k": top_k}
    if video_id:
        search_kwargs["filter"] = {"video_id": video_id}
    
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # Create BM25 retriever (sparse/keyword)
    # Note: BM25Retriever needs documents from Pinecone first
    # We'll fetch them and create BM25 index
    
    # Get all documents for this video (for BM25)
    # For simplicity, we'll get more documents than top_k
    all_docs_kwargs = {"k": top_k * 5}
    if video_id:
        all_docs_kwargs["filter"] = {"video_id": video_id}
    
    # Fetch documents for BM25
    temp_retriever = vector_store.as_retriever(search_kwargs=all_docs_kwargs)
    docs_for_bm25 = temp_retriever.invoke("initialize")  # Dummy query to get docs
    
    # Create BM25 retriever from documents
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = top_k
    
    # Combine with EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight]
    )
    
    logger.info(
        f"Created hybrid retriever: semantic ({dense_weight}) + BM25 ({sparse_weight})"
    )
    
    return ensemble_retriever