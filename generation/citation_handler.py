"""
Citation Handler for Answer Generation
Adds citations and source references to generated answers
"""

import re
from typing import List, Dict, Optional

from config.logging_config import get_logger
from utils.exceptions import CitationError

logger = get_logger(__name__)


class CitationHandler:
    """
    Handle citations in generated answers
    
    Features:
    - Extract citation markers from LLM responses
    - Link citations to source chunks
    - Format citations with metadata
    - Validate citation references
    """
    
    def __init__(self, citation_format: str = "bracket"):
        """
        Initialize citation handler
        
        Args:
            citation_format: Citation style ("bracket", "superscript", "footnote")
        """
        self.citation_format = citation_format
        logger.info(f"Initialized CitationHandler with format={citation_format}")
    
    def extract_citations(self, text: str) -> List[int]:
        """
        Extract citation chunk IDs from text
        
        Looks for patterns like [Chunk 0], [Chunk 1], etc.
        
        Args:
            text: Text containing citations
        
        Returns:
            List of cited chunk IDs
        
        Example:
            >>> handler = CitationHandler()
            >>> citations = handler.extract_citations("RAG is useful [Chunk 0]. It helps [Chunk 2].")
            >>> print(citations)
            [0, 2]
        """
        # Pattern: [Chunk N] where N is a number
        pattern = r'\[Chunk\s+(\d+)\]'
        matches = re.findall(pattern, text)
        
        # Convert to integers
        chunk_ids = [int(match) for match in matches]
        
        logger.debug(f"Extracted {len(chunk_ids)} citations: {chunk_ids}")
        
        return chunk_ids
    
    def add_source_info(
        self,
        answer: str,
        chunks: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Add source information to answer
        
        Args:
            answer: Generated answer (may contain citations)
            chunks: Original chunks used for context
        
        Returns:
            Dictionary with answer and source information
        
        Example:
            >>> handler = CitationHandler()
            >>> result = handler.add_source_info(
            ...     answer="RAG is useful [Chunk 0]",
            ...     chunks=[{"text": "RAG combines...", "chunk_id": 0, "video_id": "abc"}]
            ... )
            >>> print(result['sources'])
        """
        # Extract cited chunk IDs
        cited_ids = self.extract_citations(answer)
        
        # Create chunk ID to chunk mapping
        chunk_map = {chunk.get("chunk_id", i): chunk for i, chunk in enumerate(chunks)}
        
        # Get source information for cited chunks
        sources = []
        for chunk_id in cited_ids:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                sources.append({
                    "chunk_id": chunk_id,
                    "text": chunk.get("text", ""),
                    "video_id": chunk.get("video_id", "unknown"),
                    "score": chunk.get("score", 0.0)
                })
            else:
                logger.warning(f"Citation refers to non-existent chunk: {chunk_id}")
        
        result = {
            "answer": answer,
            "citations": cited_ids,
            "sources": sources,
            "num_citations": len(cited_ids),
            "num_valid_citations": len(sources)
        }
        
        logger.info(
            f"Added source info: {len(cited_ids)} citations, "
            f"{len(sources)} valid sources"
        )
        
        return result
    
    def format_citations(
        self,
        answer: str,
        chunks: List[Dict[str, any]],
        include_full_sources: bool = True
    ) -> str:
        """
        Format answer with enhanced citation information
        
        Args:
            answer: Answer with citations like [Chunk 0]
            chunks: Source chunks
            include_full_sources: Whether to append full source list
        
        Returns:
            Formatted answer with citations
        
        Example:
            >>> handler = CitationHandler()
            >>> formatted = handler.format_citations(
            ...     answer="RAG is useful [Chunk 0]",
            ...     chunks=chunks
            ... )
        """
        formatted_answer = answer
        
        if include_full_sources:
            # Extract sources
            source_info = self.add_source_info(answer, chunks)
            
            if source_info["sources"]:
                formatted_answer += "\n\n---\nSources:\n"
                
                for source in source_info["sources"]:
                    formatted_answer += f"\n[Chunk {source['chunk_id']}] "
                    formatted_answer += f"(Video: {source['video_id']}, "
                    formatted_answer += f"Relevance: {source['score']:.2f})\n"
                    formatted_answer += f"{source['text'][:150]}...\n"
        
        return formatted_answer
    
    def validate_citations(
        self,
        answer: str,
        chunks: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Validate that all citations reference valid chunks
        
        Args:
            answer: Answer with citations
            chunks: Available chunks
        
        Returns:
            Validation results
        
        Example:
            >>> handler = CitationHandler()
            >>> validation = handler.validate_citations(answer, chunks)
            >>> if not validation['all_valid']:
            >>>     print("Warning: Invalid citations found!")
        """
        cited_ids = self.extract_citations(answer)
        available_ids = [chunk.get("chunk_id", i) for i, chunk in enumerate(chunks)]
        
        invalid_citations = [cid for cid in cited_ids if cid not in available_ids]
        
        validation = {
            "total_citations": len(cited_ids),
            "valid_citations": len(cited_ids) - len(invalid_citations),
            "invalid_citations": invalid_citations,
            "all_valid": len(invalid_citations) == 0
        }
        
        if invalid_citations:
            logger.warning(f"Invalid citations found: {invalid_citations}")
        else:
            logger.debug("All citations valid")
        
        return validation
    
    def remove_citations(self, text: str) -> str:
        """
        Remove citation markers from text
        
        Args:
            text: Text with citations
        
        Returns:
            Text without citations
        
        Example:
            >>> handler = CitationHandler()
            >>> clean = handler.remove_citations("RAG is useful [Chunk 0].")
            >>> print(clean)
            "RAG is useful."
        """
        # Remove [Chunk N] patterns
        pattern = r'\s*\[Chunk\s+\d+\]\s*'
        clean_text = re.sub(pattern, ' ', text)
        
        # Clean up extra spaces
        clean_text = ' '.join(clean_text.split())
        
        return clean_text


def generate_answer_with_citations(
    question: str,
    chunks: List[Dict[str, any]],
    temperature: float = 0.2,
    max_tokens: int = 500
) -> Dict[str, any]:
    """
    Complete function: retrieve, generate, and add citations
    
    Args:
        question: User question
        chunks: Retrieved chunks
        temperature: LLM temperature
        max_tokens: Max response tokens
    
    Returns:
        Dictionary with answer, citations, and sources
    
    Example:
        >>> from retrieval.simple_retriever import SimpleRetriever
        >>> 
        >>> retriever = SimpleRetriever()
        >>> chunks = retriever.retrieve("What is RAG?")
        >>> result = generate_answer_with_citations("What is RAG?", chunks)
        >>> 
        >>> print(result['answer'])
        >>> print(result['sources'])
    """
    from generation.llm_client import LLMClient
    from augmentation.prompt_templates import build_qa_prompt
    
    # Build prompt with citations enabled
    prompt = build_qa_prompt(question, chunks, include_citations=True)
    
    # Generate answer
    client = LLMClient(temperature=temperature, max_tokens=max_tokens)
    answer = client.generate(prompt)
    
    # Add source information
    handler = CitationHandler()
    result = handler.add_source_info(answer, chunks)
    
    logger.info(
        f"Generated answer with {result['num_citations']} citations "
        f"({result['num_valid_citations']} valid)"
    )
    
    return result