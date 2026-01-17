"""
Indexing Chain - Complete Pipeline for Video Indexing
Orchestrates the full indexing workflow from video ID to vector storage
"""

from typing import Dict, Optional
from datetime import datetime

from indexing.document_loader import YouTubeTranscriptLoader
from indexing.text_splitter import TranscriptTextSplitter
from indexing.embeddings import EmbeddingGenerator
from indexing.vector_store import PineconeVectorStore
from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import IndexingError
from utils.validators import validate_youtube_video_id

logger = get_logger(__name__)


class IndexingChain:
    """
    Complete indexing pipeline
    
    Pipeline:
    1. Load YouTube transcript
    2. Split into chunks
    3. Generate embeddings
    4. Store in Pinecone
    
    One method call handles entire indexing process.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_dimensions: Optional[int] = None
    ):
        """
        Initialize indexing chain
        
        Args:
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            embedding_dimensions: Override default dimensions
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.embedding_dimensions = embedding_dimensions or settings.EMBEDDING_DIMENSIONS
        
        # Initialize components
        self.document_loader = YouTubeTranscriptLoader()
        self.text_splitter = TranscriptTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(
            dimensions=self.embedding_dimensions
        )
        self.vector_store = PineconeVectorStore()
        
        logger.info(
            f"Initialized IndexingChain with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, dimensions={self.embedding_dimensions}"
        )
    
    def index_video(
        self,
        video_id: str,
        namespace: str = ""
    ) -> Dict[str, any]:
        """
        Index a complete YouTube video
        
        Executes full pipeline:
        1. Fetch transcript
        2. Split into chunks
        3. Generate embeddings
        4. Store in Pinecone
        
        Args:
            video_id: YouTube video ID (11 characters)
            namespace: Pinecone namespace (default: "")
        
        Returns:
            Dictionary with indexing results and statistics
        
        Raises:
            IndexingError: If any step fails
        
        Example:
            >>> chain = IndexingChain()
            >>> result = chain.index_video("Gfr50f6ZBvo")
            >>> print(f"Indexed {result['num_chunks']} chunks")
        """
        # Validate video ID
        video_id = validate_youtube_video_id(video_id)
        
        logger.info(f"Starting indexing pipeline for video: {video_id}")
        start_time = datetime.now()
        
        try:
            # Step 1: Load transcript
            logger.info("Step 1/4: Loading transcript...")
            transcript = self.document_loader.load(video_id)
            logger.info(
                f"Loaded transcript: {transcript['num_segments']} segments, "
                f"{transcript['total_chars']} characters"
            )
            
            # Step 2: Split into chunks
            logger.info("Step 2/4: Splitting into chunks...")
            chunks = self.text_splitter.split_transcript(transcript)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            embedded_chunks = self.embedding_generator.embed_chunks(
                chunks,
                show_progress=True
            )
            logger.info(f"Generated {len(embedded_chunks)} embeddings")
            
            # Step 4: Store in Pinecone
            logger.info("Step 4/4: Storing in Pinecone...")
            store_result = self.vector_store.upsert_chunks(
                embedded_chunks,
                namespace=namespace
            )
            logger.info(f"Stored {store_result['upserted_count']} vectors")
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "video_id": video_id,
                "status": "success",
                "num_chunks": len(chunks),
                "num_embeddings": len(embedded_chunks),
                "num_stored": store_result['upserted_count'],
                "transcript_chars": transcript['total_chars'],
                "duration_seconds": duration,
                "namespace": namespace,
                "timestamp": end_time.isoformat()
            }
            
            logger.info(
                f"Indexing complete for {video_id}: {len(chunks)} chunks "
                f"in {duration:.2f} seconds"
            )
            
            return result
        
        except Exception as e:
            error_msg = f"Indexing failed for video {video_id}: {str(e)}"
            logger.error(error_msg)
            raise IndexingError(error_msg)
    
    def check_if_indexed(
        self,
        video_id: str,
        namespace: str = ""
    ) -> Dict[str, any]:
        """
        Check if a video is already indexed
        
        Args:
            video_id: YouTube video ID
            namespace: Pinecone namespace
        
        Returns:
            Dictionary with indexing status
        
        Example:
            >>> chain = IndexingChain()
            >>> status = chain.check_if_indexed("Gfr50f6ZBvo")
            >>> if status['is_indexed']:
            >>>     print("Video already indexed!")
        """
        video_id = validate_youtube_video_id(video_id)
        
        try:
            # Query Pinecone for any vectors with this video_id
            # Use a dummy embedding (won't affect results due to filter)
            dummy_embedding = [0.0] * self.embedding_dimensions
            
            results = self.vector_store.query(
                query_embedding=dummy_embedding,
                top_k=1,
                namespace=namespace,
                filter={"video_id": video_id}
            )
            
            is_indexed = len(results) > 0
            
            return {
                "video_id": video_id,
                "is_indexed": is_indexed,
                "num_chunks": len(results) if is_indexed else 0
            }
        
        except Exception as e:
            logger.error(f"Failed to check index status for {video_id}: {e}")
            return {
                "video_id": video_id,
                "is_indexed": False,
                "error": str(e)
            }
    
    def delete_video_index(
        self,
        video_id: str,
        namespace: str = ""
    ) -> Dict[str, any]:
        """
        Delete all indexed data for a video
        
        Args:
            video_id: YouTube video ID
            namespace: Pinecone namespace
        
        Returns:
            Deletion result
        
        Example:
            >>> chain = IndexingChain()
            >>> result = chain.delete_video_index("Gfr50f6ZBvo")
            >>> print(result['deleted'])
        """
        video_id = validate_youtube_video_id(video_id)
        
        logger.info(f"Deleting index for video: {video_id}")
        
        try:
            result = self.vector_store.delete_by_video_id(
                video_id=video_id,
                namespace=namespace
            )
            
            logger.info(f"Deleted vectors for video: {video_id}")
            
            return result
        
        except Exception as e:
            error_msg = f"Failed to delete index for {video_id}: {e}"
            logger.error(error_msg)
            raise IndexingError(error_msg)


def index_youtube_video(
    video_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, any]:
    """
    Convenience function to index a YouTube video
    
    Args:
        video_id: YouTube video ID
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
    
    Returns:
        Indexing results
    
    Example:
        >>> result = index_youtube_video("Gfr50f6ZBvo")
        >>> print(f"Indexed {result['num_chunks']} chunks in {result['duration_seconds']:.1f}s")
    """
    chain = IndexingChain(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return chain.index_video(video_id)