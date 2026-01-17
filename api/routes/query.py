"""
Query Endpoints
API routes for question-answering operations
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from datetime import datetime
import json

from api.models import QueryRequest, QueryResponse, ErrorResponse
from chains.qa_chain import QAChain
from config.logging_config import get_logger
from utils.exceptions import SearchError, LLMError, RAGException

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """
    Ask a question about video content
    
    Uses RAG pipeline to retrieve relevant chunks and generate answer.
    
    Args:
        request: QueryRequest with question and parameters
    
    Returns:
        QueryResponse with answer and sources
    
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    
    Example:
        POST /query
        {
            "question": "What is a language model?",
            "video_id": "Gfr50f6ZBvo",
            "retriever_type": "hybrid",
            "top_k": 4,
            "include_citations": true
        }
        
        Response:
        {
            "question": "What is a language model?",
            "answer": "A language model is...",
            "citations": [0, 1],
            "sources": [...],
            "retriever_type": "hybrid",
            "duration_seconds": 2.89
        }
    """
    logger.info(f"Query request: '{request.question[:50]}...'")
    
    try:
        # Initialize QA chain
        chain = QAChain(
            retriever_type=request.retriever_type,
            top_k=request.top_k,
            include_citations=request.include_citations,
            temperature=request.temperature or 0.2,
            max_tokens=500
        )
        
        # Execute QA pipeline
        result = chain.answer(
            question=request.question,
            video_id=request.video_id,
            top_k=request.top_k
        )
        
        # Convert to response model
        # Add timestamp if not present
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        
        response = QueryResponse(**result)
        
        logger.info(
            f"Query successful: {result['retrieved_chunks']} chunks, "
            f"{result['duration_seconds']:.2f}s"
        )
        
        return response
    
    except SearchError as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )
    
    except LLMError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(e)}"
        )
    
    except RAGException as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.post("/stream")
async def query_video_streaming(request: QueryRequest):
    """
    Ask a question with streaming response
    
    Returns answer chunks as they're generated for real-time display.
    
    Args:
        request: QueryRequest
    
    Returns:
        StreamingResponse with SSE (Server-Sent Events)
    
    Example:
        POST /query/stream
        {
            "question": "What is AI?",
            "retriever_type": "simple"
        }
        
        Response (streamed):
        data: AI
        data:  is
        data:  artificial
        data:  intelligence
        ...
    """
    logger.info(f"Streaming query: '{request.question[:50]}...'")
    
    try:
        # Initialize QA chain
        chain = QAChain(
            retriever_type=request.retriever_type,
            top_k=request.top_k,
            include_citations=False,  # Citations don't work well with streaming
            temperature=request.temperature or 0.2
        )
        
        # Generator function for streaming
        async def generate():
            try:
                for chunk in chain.answer_streaming(
                    question=request.question,
                    video_id=request.video_id,
                    top_k=request.top_k
                ):
                    # Send as Server-Sent Events format
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send completion marker
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )