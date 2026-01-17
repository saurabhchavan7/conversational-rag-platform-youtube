"""
Base Retriever Abstract Class
Defines the interface for all retriever implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers
    
    All retriever implementations must inherit from this class
    and implement the retrieve() method.
    
    Features:
    - Standard interface for all retrievers
    - Consistent input/output format
    - Logging support
    - Error handling framework
    """
    
    def __init__(self, top_k: int = 4):
        """
        Initialize base retriever
        
        Args:
            top_k: Number of results to retrieve (default: 4)
        """
        self.top_k = top_k
        logger.info(f"Initialized {self.__class__.__name__} with top_k={top_k}")
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query
        
        This is an abstract method that must be implemented by all subclasses.
        
        Args:
            query: User's search query
            top_k: Number of results to return (overrides self.top_k if provided)
            filter: Optional metadata filter (e.g., {"video_id": "abc123"})
        
        Returns:
            List of retrieved documents with scores and metadata
            Format:
            [
                {
                    "text": "chunk text...",
                    "score": 0.89,
                    "video_id": "abc123",
                    "chunk_id": 0,
                    "metadata": {...}
                },
                ...
            ]
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement retrieve() method")
    
    def _validate_query(self, query: str) -> None:
        """
        Validate query input
        
        Args:
            query: User query to validate
        
        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
    
    def _get_top_k(self, top_k: Optional[int] = None) -> int:
        """
        Get top_k value (use parameter if provided, else use instance value)
        
        Args:
            top_k: Optional top_k override
        
        Returns:
            Top_k value to use
        """
        return top_k if top_k is not None else self.top_k
    
    def _format_results(
        self,
        results: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Format retrieval results to standard format
        
        Args:
            results: Raw results from retriever
        
        Returns:
            Standardized results
        """
        formatted = []
        
        for result in results:
            formatted_result = {
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "video_id": result.get("video_id", "unknown"),
                "chunk_id": result.get("chunk_id", 0),
                "metadata": result.get("metadata", {})
            }
            formatted.append(formatted_result)
        
        return formatted
    
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Retrieve documents and return with metadata
        
        Args:
            query: User query
            top_k: Number of results
            filter: Metadata filter
        
        Returns:
            Dictionary with results and metadata:
            {
                "query": "original query",
                "results": [...],
                "num_results": 4,
                "retriever_type": "SimpleRetriever"
            }
        """
        # Validate query
        self._validate_query(query)
        
        # Retrieve
        results = self.retrieve(query, top_k, filter)
        
        # Format response
        return {
            "query": query,
            "results": results,
            "num_results": len(results),
            "retriever_type": self.__class__.__name__
        }