"""
QA Chain - Complete Pipeline for Question Answering
Orchestrates retrieval, augmentation, and generation for answering questions
"""

from typing import Dict, List, Optional
from datetime import datetime

from retrieval.simple_retriever import SimpleRetriever
from retrieval.query_rewriter import QueryRewritingRetriever
from retrieval.hybrid_retriever import HybridRetriever
from augmentation.prompt_templates import build_qa_prompt
from generation.llm_client import LLMClient
from generation.citation_handler import CitationHandler
from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import RAGException

logger = get_logger(__name__)


class QAChain:
    """
    Complete question-answering pipeline
    
    Pipeline:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Generate answer with LLM
    4. Add citations and sources
    
    Supports multiple retriever strategies and configurations.
    """
    
    def __init__(
        self,
        retriever_type: str = "simple",
        top_k: int = 4,
        include_citations: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 500
    ):
        """
        Initialize QA chain
        
        Args:
            retriever_type: "simple", "rewriting", or "hybrid"
            top_k: Number of chunks to retrieve
            include_citations: Whether to request citations
            temperature: LLM temperature
            max_tokens: Max tokens in response
        """
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.include_citations = include_citations
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize retriever based on type
        if retriever_type == "simple":
            self.retriever = SimpleRetriever(top_k=top_k)
        elif retriever_type == "rewriting":
            self.retriever = QueryRewritingRetriever(top_k=top_k)
        elif retriever_type == "hybrid":
            self.retriever = HybridRetriever(top_k=top_k)
        else:
            raise ValueError(
                f"Invalid retriever_type: {retriever_type}. "
                "Must be 'simple', 'rewriting', or 'hybrid'"
            )
        
        # Initialize generation components
        self.llm_client = LLMClient(
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.citation_handler = CitationHandler()
        
        logger.info(
            f"Initialized QAChain with retriever={retriever_type}, "
            f"top_k={top_k}, citations={include_citations}"
        )
    
    def answer(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User's question
            video_id: Optional - filter to specific video
            top_k: Optional - override top_k
        
        Returns:
            Dictionary with answer, sources, and metadata
        
        Example:
            >>> chain = QAChain(retriever_type="hybrid")
            >>> result = chain.answer("What is a language model?")
            >>> print(result['answer'])
            >>> print(result['sources'])
        """
        logger.info(f"Answering question: '{question[:50]}...'")
        start_time = datetime.now()
        
        try:
            k = top_k or self.top_k
            
            # Step 1: Retrieve relevant chunks
            logger.info(f"Step 1/3: Retrieving chunks (retriever={self.retriever_type})...")
            
            if video_id:
                # Filter to specific video
                filter = {"video_id": video_id}
                chunks = self.retriever.retrieve(question, top_k=k, filter=filter)
            else:
                # Search across all videos
                chunks = self.retriever.retrieve(question, top_k=k)
            
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            if not chunks:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information to answer this question.",
                    "sources": [],
                    "num_sources": 0,
                    "retriever_type": self.retriever_type
                }
            
            # Step 2: Build prompt
            logger.info("Step 2/3: Building prompt...")
            prompt = build_qa_prompt(
                question=question,
                chunks=chunks,
                include_citations=self.include_citations
            )
            
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Step 3: Generate answer
            logger.info("Step 3/3: Generating answer...")
            answer = self.llm_client.generate(prompt)
            
            logger.info(f"Generated answer: {len(answer)} characters")
            
            # Step 4: Process citations (if enabled)
            if self.include_citations:
                citation_info = self.citation_handler.add_source_info(answer, chunks)
                
                result = {
                    "question": question,
                    "answer": citation_info['answer'],
                    "citations": citation_info['citations'],
                    "sources": citation_info['sources'],
                    "num_sources": citation_info['num_valid_citations'],
                    "retrieved_chunks": len(chunks),
                    "retriever_type": self.retriever_type
                }
            else:
                result = {
                    "question": question,
                    "answer": answer,
                    "retrieved_chunks": len(chunks),
                    "retriever_type": self.retriever_type
                }
            
            # Add metadata
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result["duration_seconds"] = duration
            result["timestamp"] = end_time.isoformat()
            
            logger.info(f"QA complete in {duration:.2f} seconds")
            
            return result
        
        except Exception as e:
            error_msg = f"QA pipeline failed for question '{question[:50]}...': {str(e)}"
            logger.error(error_msg)
            raise RAGException(error_msg)
    
    def answer_streaming(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        """
        Answer question with streaming response
        
        Args:
            question: User's question
            video_id: Optional video filter
            top_k: Optional top_k override
        
        Yields:
            Answer chunks as they're generated
        
        Example:
            >>> chain = QAChain()
            >>> for chunk in chain.answer_streaming("What is RAG?"):
            >>>     print(chunk, end="", flush=True)
        """
        logger.info(f"Answering with streaming: '{question[:50]}...'")
        
        try:
            k = top_k or self.top_k
            
            # Retrieve chunks
            if video_id:
                filter = {"video_id": video_id}
                chunks = self.retriever.retrieve(question, top_k=k, filter=filter)
            else:
                chunks = self.retriever.retrieve(question, top_k=k)
            
            if not chunks:
                yield "I couldn't find any relevant information to answer this question."
                return
            
            # Build prompt
            prompt = build_qa_prompt(
                question=question,
                chunks=chunks,
                include_citations=self.include_citations
            )
            
            # Stream answer
            for chunk in self.llm_client.generate_streaming(prompt):
                yield chunk
        
        except Exception as e:
            logger.error(f"Streaming QA failed: {e}")
            yield f"Error: {str(e)}"


def ask_question(
    question: str,
    video_id: Optional[str] = None,
    retriever_type: str = "hybrid",
    include_citations: bool = True
) -> Dict[str, any]:
    """
    Convenience function to ask a question
    
    Args:
        question: User's question
        video_id: Optional video filter
        retriever_type: "simple", "rewriting", or "hybrid"
        include_citations: Whether to include citations
    
    Returns:
        Answer with sources
    
    Example:
        >>> result = ask_question(
        ...     "What is a language model?",
        ...     retriever_type="hybrid",
        ...     include_citations=True
        ... )
        >>> print(result['answer'])
        >>> for source in result['sources']:
        ...     print(f"Source: {source['text'][:100]}")
    """
    chain = QAChain(
        retriever_type=retriever_type,
        include_citations=include_citations
    )
    
    return chain.answer(question, video_id=video_id)