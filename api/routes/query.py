"""
Query Endpoints - API routes for question answering
"""

from fastapi import APIRouter, HTTPException, status

from api.models import QueryRequest, QueryResponse
from chains.qa_chain import answer_question
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """
    Ask a question about video content
    
    Executes complete RAG pipeline using LangChain.
    
    Example:
        POST /query
        {
            "question": "What is deep learning?",
            "video_id": "O5xeyoRL95U",
            "retriever_type": "simple",
            "top_k": 4,
            "include_citations": true
        }
        
        Response:
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is...",
            "citations": [0, 1],
            "sources": [...],
            "retrieved_chunks": 4,
            "retriever_type": "simple",
            "duration_seconds": 2.5
        }
    """
    logger.info(f"Query request: '{request.question[:50]}...'")
    
    try:
        # Execute QA pipeline
        result = answer_question(
            question=request.question,
            video_id=request.video_id,
            retriever_type=request.retriever_type,
            include_citations=request.include_citations,
            top_k=request.top_k
        )
        
        # Convert to response model
        response = QueryResponse(**result)
        
        logger.info(
            f"Query successful: {result.get('retrieved_chunks', 0)} chunks, "
            f"{result.get('duration_seconds', 0):.2f}s"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )