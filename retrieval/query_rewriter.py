"""
Query Rewriter using LLM
Improves user queries before retrieval using language models
"""

from typing import List, Dict, Optional
from openai import OpenAI

from retrieval.base_retriever import BaseRetriever
from retrieval.simple_retriever import SimpleRetriever
from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import QueryRewriteError

logger = get_logger(__name__)


# Prompt template for query rewriting
QUERY_REWRITE_PROMPT = """You are an expert at reformulating user queries to improve search results.

Given a user's query, rewrite it to be more specific, clear, and optimized for semantic search.

Rules:
1. Expand abbreviations (e.g., "ML" -> "machine learning")
2. Add context if the query is vague
3. Fix typos and grammar
4. Make it a complete question or statement
5. Keep the core intent of the original query
6. Output ONLY the rewritten query, nothing else

Examples:
User query: "what's rag"
Rewritten: "What is Retrieval-Augmented Generation and how does it work?"

User query: "transformr model"
Rewritten: "What is a transformer model in machine learning?"

User query: "train llm"
Rewritten: "How are large language models trained?"

Now rewrite this query:
User query: {query}
Rewritten:"""


class QueryRewritingRetriever(BaseRetriever):
    """
    Retriever with LLM-based query rewriting
    
    Workflow:
    1. User query -> LLM rewriting -> Improved query
    2. Improved query -> Simple retrieval
    3. Return results
    
    Benefits:
    - Handles vague queries ("what's that thing")
    - Fixes typos and grammar
    - Expands abbreviations
    - Adds missing context
    
    Typical improvement: 5-10% better retrieval accuracy
    """
    
    def __init__(
        self,
        top_k: int = 4,
        base_retriever: Optional[SimpleRetriever] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize query rewriting retriever
        
        Args:
            top_k: Number of results to retrieve
            base_retriever: Underlying retriever (creates SimpleRetriever if None)
            llm_model: LLM for rewriting (default: from settings)
            temperature: LLM temperature (0.0 = deterministic)
        """
        super().__init__(top_k=top_k)
        
        # Initialize base retriever
        self.base_retriever = base_retriever or SimpleRetriever(top_k=top_k)
        
        # Initialize LLM client
        self.llm_model = llm_model or settings.OPENAI_CHAT_MODEL
        self.temperature = temperature
        
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                f"Initialized QueryRewritingRetriever with model={self.llm_model}, "
                f"temperature={self.temperature}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise QueryRewriteError(f"LLM initialization failed: {e}")
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query using LLM
        
        Args:
            query: Original user query
        
        Returns:
            Rewritten, improved query
        
        Raises:
            QueryRewriteError: If rewriting fails
        
        Example:
            >>> rewriter = QueryRewritingRetriever()
            >>> improved = rewriter.rewrite_query("what's rag")
            >>> print(improved)
            "What is Retrieval-Augmented Generation and how does it work?"
        """
        logger.info(f"Rewriting query: '{query}'")
        
        try:
            # Format prompt
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=100  # Queries should be short
            )
            
            # Extract rewritten query
            rewritten_query = response.choices[0].message.content.strip()
            
            logger.info(f"Rewritten query: '{rewritten_query}'")
            
            return rewritten_query
        
        except Exception as e:
            error_msg = f"Failed to rewrite query '{query}': {str(e)}"
            logger.error(error_msg)
            
            # Fallback: use original query
            logger.warning("Falling back to original query")
            return query
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None,
        skip_rewrite: bool = False
    ) -> List[Dict[str, any]]:
        """
        Retrieve with query rewriting
        
        Args:
            query: User's search query
            top_k: Number of results
            filter: Metadata filter
            skip_rewrite: If True, skip rewriting (use original query)
        
        Returns:
            Retrieved chunks
        
        Example:
            >>> retriever = QueryRewritingRetriever()
            >>> results = retriever.retrieve("what's a transformer")
            >>> # Query is rewritten, then searched
        """
        # Validate query
        self._validate_query(query)
        
        # Get top_k
        k = self._get_top_k(top_k)
        
        # Rewrite query (unless skipped)
        if not skip_rewrite:
            rewritten_query = self.rewrite_query(query)
        else:
            rewritten_query = query
            logger.info("Skipping query rewriting (skip_rewrite=True)")
        
        # Use base retriever with rewritten query
        logger.info(f"Retrieving with rewritten query (top_k={k})")
        results = self.base_retriever.retrieve(
            query=rewritten_query,
            top_k=k,
            filter=filter
        )
        
        # Add original query to results metadata
        for result in results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
        
        return results
    
    def compare_queries(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """
        Compare retrieval results with and without query rewriting
        
        Useful for debugging and evaluation.
        
        Args:
            query: User query
            top_k: Number of results
        
        Returns:
            Dictionary with both results for comparison
        
        Example:
            >>> retriever = QueryRewritingRetriever()
            >>> comparison = retriever.compare_queries("what's rag", top_k=3)
            >>> print("Original:", comparison['original_results'])
            >>> print("Rewritten:", comparison['rewritten_results'])
        """
        logger.info(f"Comparing retrieval with/without rewriting for: '{query}'")
        
        # Get results without rewriting
        original_results = self.retrieve(query, top_k=top_k, skip_rewrite=True)
        
        # Get results with rewriting
        rewritten_results = self.retrieve(query, top_k=top_k, skip_rewrite=False)
        
        # Extract rewritten query
        rewritten_query = rewritten_results[0].get("rewritten_query", query) if rewritten_results else query
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "original_results": original_results,
            "rewritten_results": rewritten_results,
            "improvement": self._calculate_improvement(original_results, rewritten_results)
        }
    
    def _calculate_improvement(
        self,
        original_results: List[Dict],
        rewritten_results: List[Dict]
    ) -> Dict[str, any]:
        """
        Calculate improvement metrics between original and rewritten
        
        Args:
            original_results: Results from original query
            rewritten_results: Results from rewritten query
        
        Returns:
            Improvement statistics
        """
        if not original_results or not rewritten_results:
            return {"avg_score_diff": 0.0}
        
        # Calculate average scores
        original_avg = sum(r.get("score", 0) for r in original_results) / len(original_results)
        rewritten_avg = sum(r.get("score", 0) for r in rewritten_results) / len(rewritten_results)
        
        return {
            "original_avg_score": original_avg,
            "rewritten_avg_score": rewritten_avg,
            "avg_score_diff": rewritten_avg - original_avg,
            "improvement_pct": ((rewritten_avg - original_avg) / original_avg * 100) if original_avg > 0 else 0
        }


def retrieve_with_rewriting(
    query: str,
    top_k: int = 4,
    video_id: Optional[str] = None
) -> List[Dict[str, any]]:
    """
    Convenience function for retrieval with query rewriting
    
    Args:
        query: User query
        top_k: Number of results
        video_id: Optional video filter
    
    Returns:
        Retrieved chunks
    
    Example:
        >>> results = retrieve_with_rewriting("what's ml", top_k=5)
        >>> print(results[0]['rewritten_query'])
        "What is machine learning and how does it work?"
    """
    retriever = QueryRewritingRetriever(top_k=top_k)
    
    filter = {"video_id": video_id} if video_id else None
    
    return retriever.retrieve(query, top_k=top_k, filter=filter)