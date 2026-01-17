"""
Prompt Templates for RAG System
Contains all system prompts for answer generation
"""

from typing import List, Dict
from config.logging_config import get_logger

logger = get_logger(__name__)


# ============================================
# Main QA Prompt Template
# ============================================

QA_SYSTEM_PROMPT = """You are an AI assistant helping users understand YouTube video content.

Your task is to answer questions based ONLY on the provided context from video transcripts.

Guidelines:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information in this video to answer that question"
3. Be specific and cite which part of the video the information comes from
4. Keep answers concise but complete
5. Use the same language and terminology as the video
6. Do not make up or infer information not present in the context

Context from video transcript:
{context}

User question: {question}

Answer:"""


# ============================================
# QA Prompt with Citations
# ============================================

QA_PROMPT_WITH_CITATIONS = """You are an AI assistant helping users understand YouTube video content.

Your task is to answer questions based ONLY on the provided context from video transcripts, and include citations.

Guidelines:
1. Answer ONLY using information from the provided context
2. After each fact or claim, add a citation in brackets like [Chunk 0], [Chunk 1], etc.
3. If the context doesn't contain enough information, say so clearly
4. Be specific and accurate
5. Multiple chunks can support the same point - cite all relevant chunks
6. Keep answers concise but complete

Context (with chunk IDs):
{context}

User question: {question}

Answer (with citations):"""


# ============================================
# Conversational QA Prompt (Chat History)
# ============================================

CONVERSATIONAL_QA_PROMPT = """You are an AI assistant helping users understand YouTube video content through conversation.

You have access to the video transcript context and the conversation history.

Guidelines:
1. Answer based ONLY on the provided video transcript context
2. Consider the conversation history for context
3. Maintain consistency with previous answers
4. If asked about something not in the video, say so clearly
5. Be conversational but accurate

Conversation History:
{chat_history}

Video Transcript Context:
{context}

Current Question: {question}

Answer:"""


# ============================================
# Summary Prompt
# ============================================

SUMMARY_PROMPT = """You are an AI assistant that creates concise summaries of YouTube video content.

Based on the provided transcript context, create a summary.

Guidelines:
1. Capture the main topics and key points
2. Use bullet points for clarity
3. Keep it concise (3-5 bullet points)
4. Use information ONLY from the provided context
5. Maintain the video's tone and terminology

Video Transcript Context:
{context}

Summary:"""


# ============================================
# Helper Functions
# ============================================

def format_context_for_prompt(chunks: List[Dict[str, any]], include_chunk_ids: bool = False) -> str:
    """
    Format retrieved chunks into context string for prompt
    
    Args:
        chunks: List of retrieved chunk dictionaries
        include_chunk_ids: Whether to include chunk IDs for citations
    
    Returns:
        Formatted context string
    
    Example:
        >>> chunks = [{"text": "AI is...", "chunk_id": 0}, {"text": "ML is...", "chunk_id": 1}]
        >>> context = format_context_for_prompt(chunks, include_chunk_ids=True)
        >>> print(context)
        [Chunk 0]
        AI is...
        
        [Chunk 1]
        ML is...
    """
    if not chunks:
        logger.warning("No chunks provided for context formatting")
        return "No context available."
    
    context_parts = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", 0)
        
        if include_chunk_ids:
            context_parts.append(f"[Chunk {chunk_id}]\n{text}")
        else:
            context_parts.append(text)
    
    # Join with double newlines for separation
    context = "\n\n".join(context_parts)
    
    logger.debug(f"Formatted context: {len(chunks)} chunks, {len(context)} characters")
    
    return context


def build_qa_prompt(
    question: str,
    chunks: List[Dict[str, any]],
    include_citations: bool = False
) -> str:
    """
    Build complete QA prompt from question and chunks
    
    Args:
        question: User's question
        chunks: Retrieved chunks
        include_citations: Whether to request citations in answer
    
    Returns:
        Complete prompt string ready for LLM
    
    Example:
        >>> chunks = retriever.retrieve("What is RAG?")
        >>> prompt = build_qa_prompt("What is RAG?", chunks, include_citations=True)
        >>> answer = llm.generate(prompt)
    """
    # Format context
    context = format_context_for_prompt(chunks, include_chunk_ids=include_citations)
    
    # Choose appropriate template
    if include_citations:
        template = QA_PROMPT_WITH_CITATIONS
    else:
        template = QA_SYSTEM_PROMPT
    
    # Fill template
    prompt = template.format(context=context, question=question)
    
    logger.info(
        f"Built QA prompt: question_length={len(question)}, "
        f"context_length={len(context)}, citations={include_citations}"
    )
    
    return prompt


def build_summary_prompt(chunks: List[Dict[str, any]]) -> str:
    """
    Build prompt for video summarization
    
    Args:
        chunks: Retrieved or all chunks from video
    
    Returns:
        Complete summary prompt
    
    Example:
        >>> all_chunks = get_all_chunks_for_video("abc123")
        >>> prompt = build_summary_prompt(all_chunks)
        >>> summary = llm.generate(prompt)
    """
    context = format_context_for_prompt(chunks, include_chunk_ids=False)
    prompt = SUMMARY_PROMPT.format(context=context)
    
    logger.info(f"Built summary prompt: {len(chunks)} chunks")
    
    return prompt


def build_conversational_prompt(
    question: str,
    chunks: List[Dict[str, any]],
    chat_history: List[Dict[str, str]]
) -> str:
    """
    Build prompt for conversational QA with chat history
    
    Args:
        question: Current question
        chunks: Retrieved chunks
        chat_history: Previous conversation turns
            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        Complete conversational prompt
    
    Example:
        >>> history = [
        ...     {"role": "user", "content": "What is RAG?"},
        ...     {"role": "assistant", "content": "RAG is..."}
        ... ]
        >>> prompt = build_conversational_prompt("Tell me more", chunks, history)
    """
    # Format context
    context = format_context_for_prompt(chunks, include_chunk_ids=False)
    
    # Format chat history
    history_text = ""
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_text += f"{role.capitalize()}: {content}\n"
    
    # Fill template
    prompt = CONVERSATIONAL_QA_PROMPT.format(
        chat_history=history_text,
        context=context,
        question=question
    )
    
    logger.info(f"Built conversational prompt with {len(chat_history)} history turns")
    
    return prompt


def get_token_count_estimate(text: str) -> int:
    """
    Estimate token count for text
    
    Rough approximation: 1 token ≈ 4 characters
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    
    Note:
        This is a rough estimate. For precise counting, use tiktoken library.
    """
    return len(text) // 4


def check_prompt_length(prompt: str, max_tokens: int = 3000) -> Dict[str, any]:
    """
    Check if prompt is within token limits
    
    Args:
        prompt: Complete prompt string
        max_tokens: Maximum allowed tokens
    
    Returns:
        Dictionary with length info and warning flag
    
    Example:
        >>> prompt = build_qa_prompt(question, chunks)
        >>> check = check_prompt_length(prompt)
        >>> if check['exceeds_limit']:
        >>>     print("Warning: Prompt too long!")
    """
    estimated_tokens = get_token_count_estimate(prompt)
    
    result = {
        "estimated_tokens": estimated_tokens,
        "max_tokens": max_tokens,
        "exceeds_limit": estimated_tokens > max_tokens,
        "usage_pct": (estimated_tokens / max_tokens) * 100
    }
    
    if result["exceeds_limit"]:
        logger.warning(
            f"Prompt may exceed token limit: {estimated_tokens} > {max_tokens}"
        )
    
    return result