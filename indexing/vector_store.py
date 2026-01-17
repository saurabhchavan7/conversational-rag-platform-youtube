"""
Vector Store Operations for Pinecone
Stores and retrieves embeddings from Pinecone vector database
"""

from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
import time

from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import VectorStoreError

logger = get_logger(__name__)


class PineconeVectorStore:
    """
    Pinecone vector store for embeddings
    
    Features:
    - Automatic index connection
    - Batch upsert operations
    - Metadata storage
    - Query operations
    - Index statistics
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize Pinecone vector store
        
        Args:
            index_name: Pinecone index name (default: from settings)
            dimension: Embedding dimensions (default: from settings)
        """
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSIONS
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            logger.info(f"Initialized Pinecone client for index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise VectorStoreError(f"Pinecone initialization failed: {e}")
    
    def upsert_chunks(
        self,
        chunks: List[Dict[str, any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, any]:
        """
        Upsert embedded chunks to Pinecone
        
        Args:
            chunks: List of chunks with embeddings
            namespace: Pinecone namespace (default: "")
            batch_size: Vectors per batch (default: 100)
        
        Returns:
            Dictionary with upsert statistics
        
        Raises:
            VectorStoreError: If upsert fails
        
        Example:
            >>> store = PineconeVectorStore()
            >>> result = store.upsert_chunks(embedded_chunks)
            >>> print(result['upserted_count'])
        """
        if not chunks:
            logger.warning("No chunks provided for upsert")
            return {"upserted_count": 0}
        
        logger.info(f"Upserting {len(chunks)} chunks to Pinecone")
        
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for chunk in chunks:
                if "embedding" not in chunk:
                    logger.warning(f"Chunk {chunk.get('chunk_id')} missing embedding, skipping")
                    continue
                
                # Create unique ID
                video_id = chunk.get("metadata", {}).get("video_id", "unknown")
                chunk_id = chunk.get("chunk_id", 0)
                vector_id = f"{video_id}_{chunk_id}"
                
                # Prepare metadata (Pinecone has limits on metadata size)
                metadata = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "text": chunk["text"][:1000],  # Limit text to 1000 chars
                    "language": chunk.get("metadata", {}).get("language", "unknown"),
                    "source": chunk.get("metadata", {}).get("source", "youtube")
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": chunk["embedding"],
                    "metadata": metadata
                })
            
            logger.info(f"Prepared {len(vectors)} vectors for upsert")
            
            # Upsert in batches
            upserted_count = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    self.index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                    upserted_count += len(batch)
                    logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
                    
                    # Small delay to avoid rate limits
                    if i + batch_size < len(vectors):
                        time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                    raise VectorStoreError(f"Batch upsert failed: {e}")
            
            logger.info(f"Successfully upserted {upserted_count} vectors")
            
            return {
                "upserted_count": upserted_count,
                "total_chunks": len(chunks),
                "namespace": namespace
            }
        
        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to upsert chunks: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        namespace: str = "",
        filter: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Query Pinecone for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            namespace: Pinecone namespace
            filter: Metadata filter (e.g., {"video_id": "abc123"})
        
        Returns:
            List of matches with scores and metadata
        
        Example:
            >>> store = PineconeVectorStore()
            >>> results = store.query(embedding, top_k=5)
            >>> for match in results:
            >>>     print(match['text'], match['score'])
        """
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty")
        
        try:
            logger.info(f"Querying Pinecone: top_k={top_k}, namespace='{namespace}'")
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )
            
            # Parse results
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "video_id": match.metadata.get("video_id", ""),
                    "chunk_id": match.metadata.get("chunk_id", 0),
                    "metadata": match.metadata
                })
            
            logger.info(f"Found {len(matches)} matches")
            
            return matches
        
        except Exception as e:
            error_msg = f"Failed to query Pinecone: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def delete_by_video_id(
        self,
        video_id: str,
        namespace: str = ""
    ) -> Dict[str, any]:
        """
        Delete all vectors for a specific video
        
        Args:
            video_id: YouTube video ID
            namespace: Pinecone namespace
        
        Returns:
            Dictionary with deletion info
        """
        try:
            logger.info(f"Deleting vectors for video: {video_id}")
            
            # Delete using metadata filter
            self.index.delete(
                filter={"video_id": video_id},
                namespace=namespace
            )
            
            logger.info(f"Deleted vectors for video: {video_id}")
            
            return {
                "deleted": True,
                "video_id": video_id,
                "namespace": namespace
            }
        
        except Exception as e:
            error_msg = f"Failed to delete vectors for {video_id}: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def get_index_stats(self, namespace: str = "") -> Dict[str, any]:
        """
        Get index statistics
        
        Args:
            namespace: Pinecone namespace
        
        Returns:
            Index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            logger.info(f"Index stats: {stats}")
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


def store_embeddings_in_pinecone(
    embedded_chunks: List[Dict[str, any]],
    video_id: Optional[str] = None,
    namespace: str = ""
) -> Dict[str, any]:
    """
    Convenience function to store embeddings in Pinecone
    
    Args:
        embedded_chunks: Chunks with embeddings
        video_id: Optional video ID for logging
        namespace: Pinecone namespace
    
    Returns:
        Upsert statistics
    
    Example:
        >>> from indexing.document_loader import load_youtube_transcript
        >>> from indexing.text_splitter import split_transcript_into_chunks
        >>> from indexing.embeddings import generate_embeddings_for_chunks
        >>> 
        >>> transcript = load_youtube_transcript("abc123")
        >>> chunks = split_transcript_into_chunks(transcript)
        >>> embedded_chunks = generate_embeddings_for_chunks(chunks)
        >>> result = store_embeddings_in_pinecone(embedded_chunks)
    """
    store = PineconeVectorStore()
    result = store.upsert_chunks(embedded_chunks, namespace=namespace)
    
    if video_id:
        logger.info(f"Stored {result['upserted_count']} vectors for video {video_id}")
    
    return result