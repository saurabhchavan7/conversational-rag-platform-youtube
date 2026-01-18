"""
Indexing Chain - Complete Pipeline using LangChain
Simple orchestration of LangChain components
"""

import os
from typing import Dict, Optional
from datetime import datetime
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from indexing.document_loader import load_youtube_transcript
from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import IndexingError
from utils.validators import validate_youtube_video_id

logger = get_logger(__name__)

# Set environment variables for LangChain (it reads from os.environ)
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def index_video_to_pinecone(
    video_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    namespace: str = ""
) -> Dict[str, any]:
    """
    Index a YouTube video using pure LangChain pipeline
    
    Uses LangChain for everything:
    - RecursiveCharacterTextSplitter for chunking
    - OpenAIEmbeddings for embedding generation
    - PineconeVectorStore for storage
    
    Args:
        video_id: YouTube video ID
        chunk_size: Chunk size (default: 1000)
        chunk_overlap: Overlap size (default: 200)
        namespace: Pinecone namespace
    
    Returns:
        Indexing results dictionary
    
    Example:
        >>> result = index_video_to_pinecone("O5xeyoRL95U")
        >>> print(f"Indexed {result['num_chunks']} chunks")
    """
    video_id = validate_youtube_video_id(video_id)
    
    logger.info(f"Starting LangChain indexing pipeline for video: {video_id}")
    start_time = datetime.now()
    
    try:
        # Step 1: Load transcript
        logger.info("Step 1/3: Loading transcript...")
        transcript = load_youtube_transcript(video_id)
        logger.info(f"Loaded {transcript['total_chars']} characters")
        
        # Step 2: Create LangChain text splitter
        logger.info("Step 2/3: Splitting with LangChain...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split text and create LangChain Documents
        chunks = text_splitter.create_documents(
            texts=[transcript["text"]],
            metadatas=[{
                "video_id": video_id,
                "language": transcript["language"],
                "source": "youtube"
            }]
        )
        logger.info(f"Created {len(chunks)} LangChain documents")
        
        # Add chunk_id to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
        
        # Create unique IDs
        ids = [f"{video_id}_{idx}" for idx in range(len(chunks))]
        
        # Step 3: Use LangChain to embed and store in Pinecone
        logger.info("Step 3/3: Embedding and storing with LangChain...")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # LangChain does embedding + storage in one call!
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
            namespace=namespace,
            ids=ids
        )
        
        logger.info(f"Stored {len(chunks)} vectors via LangChain")
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "video_id": video_id,
            "status": "success",
            "num_chunks": len(chunks),
            "transcript_chars": transcript['total_chars'],
            "duration_seconds": duration,
            "namespace": namespace,
            "timestamp": end_time.isoformat()
        }
        
        logger.info(f"Indexing complete: {len(chunks)} chunks in {duration:.2f}s")
        
        return result
    
    except Exception as e:
        error_msg = f"Indexing failed for {video_id}: {str(e)}"
        logger.error(error_msg)
        raise IndexingError(error_msg)


def check_if_video_indexed(video_id: str, namespace: str = "") -> bool:
    """
    Check if video is already indexed using LangChain
    
    Args:
        video_id: YouTube video ID
        namespace: Pinecone namespace
    
    Returns:
        True if indexed, False otherwise
    
    Example:
        >>> if check_if_video_indexed("O5xeyoRL95U"):
        >>>     print("Already indexed!")
    """
    video_id = validate_youtube_video_id(video_id)
    
    try:
        # Create embeddings instance
        embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Connect to existing index
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
        
        # Try to retrieve with filter
        # Use LangChain's similarity search with filter
        results = vector_store.similarity_search(
            query="test",  # Dummy query
            k=1,
            filter={"video_id": video_id}
        )
        
        return len(results) > 0
    
    except Exception as e:
        logger.error(f"Failed to check if indexed: {e}")
        return False