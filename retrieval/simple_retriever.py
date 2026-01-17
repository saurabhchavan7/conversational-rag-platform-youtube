"""
Simple Retriever using Pinecone Vector Search
Performs basic semantic similarity search
"""

from typing import List, Dict, Optional

from retrieval.base_retriever import BaseRetriever
from indexing.vector_store import PineconeVectorStore
from indexing.embeddings import EmbeddingGenerator
from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import SearchError

logger = get_logger(__name__)


class SimpleRetriever(BaseRetriever):
    """
    Simple semantic similarity retriever
    
    Workflow:
    1. Takes user query
    2. Generates query embedding
    3. Searches Pinecone for similar vectors
    4. Returns top_k most similar chunks
    
    This is the baseline retriever - other retrievers build on this.
    """
    
    def __init__(
        self,
        top_k: int = 4,
        vector_store: Optional[PineconeVectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize simple retriever
        
        Args:
            top_k: Number of results to retrieve (default: 4)
            vector_store: Pinecone vector store (creates new if None)
            embedding_generator: Embedding generator (creates new if None)
        """
        super().__init__(top_k=top_k)
        
        # Initialize components
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        logger.info(
            f"Initialized SimpleRetriever with top_k={top_k}, "
            f"vector_store={self.vector_store.index_name}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks using semantic similarity
        
        Args:
            query: User's search query
            top_k: Number of results (overrides instance value if provided)
            filter: Metadata filter (e.g., {"video_id": "abc123"})
        
        Returns:
            List of retrieved chunks with scores
            
        Raises:
            SearchError: If retrieval fails
        
        Example:
            >>> retriever = SimpleRetriever(top_k=4)
            >>> results = retriever.retrieve("What is a language model?")
            >>> print(results[0]['text'])
        """
        # Validate query
        self._validate_query(query)
        
        # Get top_k value
        k = self._get_top_k(top_k)
        
        logger.info(f"Retrieving for query: '{query[:50]}...' (top_k={k})")
        
        try:
            # Step 1: Generate query embedding
            logger.debug("Generating query embedding")
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Step 2: Search Pinecone
            logger.debug(f"Searching Pinecone with top_k={k}")
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=k,
                filter=filter
            )
            
            # Step 3: Format results
            formatted_results = self._format_results(results)
            
            logger.info(f"Retrieved {len(formatted_results)} results")
            
            # Log top result for debugging
            if formatted_results:
                logger.debug(
                    f"Top result: score={formatted_results[0]['score']:.4f}, "
                    f"text={formatted_results[0]['text'][:50]}..."
                )
            
            return formatted_results
        
        except Exception as e:
            error_msg = f"Retrieval failed for query '{query[:50]}...': {str(e)}"
            logger.error(error_msg)
            raise SearchError(error_msg)
    
    def retrieve_for_video(
        self,
        query: str,
        video_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve chunks from a specific video only
        
        Args:
            query: User query
            video_id: Filter results to this video only
            top_k: Number of results
        
        Returns:
            Retrieved chunks from specified video
        
        Example:
            >>> retriever = SimpleRetriever()
            >>> results = retriever.retrieve_for_video(
            ...     query="What is RAG?",
            ...     video_id="Gfr50f6ZBvo"
            ... )
        """
        logger.info(f"Retrieving from video {video_id} only")
        
        # Use metadata filter to restrict to specific video
        filter = {"video_id": video_id}
        
        return self.retrieve(query, top_k=top_k, filter=filter)


def search_pinecone(
    query: str,
    top_k: int = 4,
    video_id: Optional[str] = None
) -> List[Dict[str, any]]:
    """
    Convenience function for simple retrieval
    
    Args:
        query: User query
        top_k: Number of results
        video_id: Optional video filter
    
    Returns:
        Retrieved chunks
    
    Example:
        >>> results = search_pinecone("What is a transformer?", top_k=5)
        >>> print(results[0]['text'])
    """
    retriever = SimpleRetriever(top_k=top_k)
    
    if video_id:
        return retriever.retrieve_for_video(query, video_id, top_k)
    else:
        return retriever.retrieve(query, top_k)