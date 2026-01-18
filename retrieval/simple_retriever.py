"""
Simple Retriever using LangChain's Pinecone Integration
Basic semantic similarity search - the baseline retriever
"""

import os
from typing import List, Optional
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def create_simple_retriever(
    video_id: Optional[str] = None,
    top_k: int = 4,
    namespace: str = ""
):
    """
    Create a simple similarity search retriever (baseline)
    
    This is the most basic retriever - just semantic similarity search.
    
    Args:
        video_id: Optional - filter to specific video
        top_k: Number of chunks to retrieve (default: 4)
        namespace: Pinecone namespace
    
    Returns:
        LangChain VectorStoreRetriever
    
    Example:
        >>> retriever = create_simple_retriever(video_id="O5xeyoRL95U", top_k=4)
        >>> results = retriever.invoke("What is deep learning?")
    """
    logger.info(f"Creating simple retriever: video_id={video_id}, top_k={top_k}")
    
    # Create LangChain embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    # Connect to Pinecone via LangChain
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    
    # Configure search
    search_kwargs = {"k": top_k}
    if video_id:
        search_kwargs["filter"] = {"video_id": video_id}
    
    # Create retriever with similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    logger.info("Created simple similarity retriever")
    
    return retriever


def retrieve_chunks(
    question: str,
    video_id: Optional[str] = None,
    top_k: int = 4
) -> List[Document]:
    """
    Retrieve relevant chunks for a question
    
    Convenience function for simple retrieval.
    
    Args:
        question: User's question
        video_id: Optional video filter
        top_k: Number of chunks to retrieve
    
    Returns:
        List of LangChain Documents (retrieved chunks)
    
    Example:
        >>> chunks = retrieve_chunks(
        >>>     question="What is deep learning?",
        >>>     video_id="O5xeyoRL95U",
        >>>     top_k=4
        >>> )
        >>> 
        >>> for chunk in chunks:
        >>>     print(chunk.page_content[:100])
        >>>     print(f"Video: {chunk.metadata['video_id']}")
    """
    logger.info(f"Retrieving for question: '{question[:50]}...'")
    
    # Create retriever
    retriever = create_retriever(video_id=video_id, top_k=top_k)
    
    # Retrieve using LangChain
    results = retriever.invoke(question)
    
    logger.info(f"Retrieved {len(results)} chunks")
    
    return results