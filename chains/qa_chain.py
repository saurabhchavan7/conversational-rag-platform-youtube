"""
QA Chain - Complete RAG Pipeline using LangChain LCEL
Combines retrieval, augmentation, and generation into one chain
"""

from typing import Optional, Dict, Any
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from retrieval.simple_retriever import create_simple_retriever
from retrieval.query_rewriter import create_rewriting_retriever
from retrieval.hybrid_retriever import create_hybrid_retriever
from augmentation.prompt_templates import (
    QA_PROMPT,
    QA_PROMPT_WITH_CITATIONS,
    format_docs_for_prompt
)
from generation.llm_client import create_llm
from generation.citation_handler import add_source_info
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def create_qa_chain(
    video_id: str,
    retriever_type: str = "simple",
    include_citations: bool = True,
    top_k: int = 4
):
    """
    Create complete QA chain using LangChain LCEL
    
    This chain combines:
    - Retrieval (Phase 6)
    - Augmentation (Phase 7)
    - Generation (Phase 8)
    
    Into one LangChain chain that executes with single invoke()
    
    Args:
        video_id: YouTube video ID to search
        retriever_type: "simple", "rewriting", or "hybrid"
        include_citations: Whether to request citations
        top_k: Number of chunks to retrieve
    
    Returns:
        LangChain LCEL chain
    
    Example:
        >>> chain = create_qa_chain(video_id="O5xeyoRL95U", retriever_type="simple")
        >>> answer = chain.invoke("What is deep learning?")
        >>> print(answer)  # Just the answer string!
    """
    logger.info(
        f"Creating QA chain: retriever={retriever_type}, "
        f"citations={include_citations}, top_k={top_k}"
    )
    
    # Step 1: Create retriever based on type
    if retriever_type == "simple":
        retriever = create_simple_retriever(video_id=video_id, top_k=top_k)
    elif retriever_type == "rewriting":
        retriever = create_rewriting_retriever(video_id=video_id, top_k=top_k)
    elif retriever_type == "hybrid":
        retriever = create_hybrid_retriever(video_id=video_id, top_k=top_k)
    else:
        raise ValueError(f"Invalid retriever_type: {retriever_type}")
    
    # Step 2: Create context formatting function
    def format_context(docs):
        return format_docs_for_prompt(docs, include_chunk_ids=include_citations)
    
    # Step 3: Select prompt template
    prompt = QA_PROMPT_WITH_CITATIONS if include_citations else QA_PROMPT
    
    # Step 4: Create LLM
    llm = create_llm(temperature=0.2, max_tokens=500)
    
    # Step 5: Build LangChain LCEL chain
    # This is the magic - everything runs automatically!
    qa_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("Created complete QA chain using LCEL")
    
    return qa_chain


def answer_question(
    question: str,
    video_id: str,
    retriever_type: str = "simple",
    include_citations: bool = True,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Answer a question about a video (complete RAG pipeline)
    
    This is the main function - executes complete pipeline:
    1. Retrieve relevant chunks (Phase 6)
    2. Format into prompt (Phase 7)
    3. Generate answer (Phase 8)
    4. Extract citations (Phase 8)
    
    Args:
        question: User's question
        video_id: Video to search
        retriever_type: Which retrieval strategy
        include_citations: Request citations in answer
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with answer, citations, sources, and metadata
    
    Example:
        >>> result = answer_question(
        >>>     question="What is deep learning?",
        >>>     video_id="O5xeyoRL95U",
        >>>     retriever_type="hybrid",
        >>>     include_citations=True
        >>> )
        >>> print(result['answer'])
        >>> print(result['citations'])
    """
    logger.info(f"Answering question: '{question[:50]}...'")
    start_time = datetime.now()
    
    try:
        # Create retriever
        if retriever_type == "simple":
            retriever = create_simple_retriever(video_id=video_id, top_k=top_k)
        elif retriever_type == "rewriting":
            retriever = create_rewriting_retriever(video_id=video_id, top_k=top_k)
        elif retriever_type == "hybrid":
            retriever = create_hybrid_retriever(video_id=video_id, top_k=top_k)
        else:
            retriever = create_simple_retriever(video_id=video_id, top_k=top_k)
        
        # Retrieve documents
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} chunks")
        
        # Format context
        context = format_docs_for_prompt(docs, include_chunk_ids=include_citations)
        
        # Select prompt
        prompt = QA_PROMPT_WITH_CITATIONS if include_citations else QA_PROMPT
        prompt_text = prompt.format(context=context, question=question)
        
        # Generate answer
        llm = create_llm(temperature=0.2, max_tokens=500)
        response = llm.invoke(prompt_text)
        answer = response.content
        
        logger.info(f"Generated answer: {len(answer)} characters")
        
        # Process citations if enabled
        if include_citations:
            result = add_source_info(answer, docs)
            result['retrieved_chunks'] = len(docs)
            result['retriever_type'] = retriever_type
        else:
            result = {
                "answer": answer,
                "retrieved_chunks": len(docs),
                "retriever_type": retriever_type
            }
        
        # Add metadata
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result['duration_seconds'] = duration
        result['timestamp'] = end_time.isoformat()
        result['question'] = question
        
        logger.info(f"QA complete in {duration:.2f} seconds")
        
        return result
    
    except Exception as e:
        logger.error(f"QA pipeline failed: {e}")
        raise