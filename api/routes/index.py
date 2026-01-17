"""
Indexing Endpoints
API routes for video indexing operations
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from api.models import (
    IndexRequest,
    IndexResponse,
    VideoStatusResponse,
    DeleteRequest,
    DeleteResponse,
    ErrorResponse
)
from chains.indexing_chain import IndexingChain
from config.logging_config import get_logger
from utils.exceptions import IndexingError, InvalidVideoIDError

logger = get_logger(__name__)

router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_video(request: IndexRequest):
    """
    Index a YouTube video
    
    Creates searchable index from video transcript.
    
    Args:
        request: IndexRequest with video_id and optional parameters
    
    Returns:
        IndexResponse with indexing results
    
    Raises:
        HTTPException: 400 for invalid input, 500 for indexing errors
    
    Example:
        POST /index
        {
            "video_id": "Gfr50f6ZBvo",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    """
    logger.info(f"Indexing request for video: {request.video_id}")
    
    try:
        # Initialize indexing chain
        chain = IndexingChain(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Execute indexing
        result = chain.index_video(
            video_id=request.video_id,
            namespace=request.namespace
        )
        
        # Convert to response model
        response = IndexResponse(**result)
        
        logger.info(
            f"Successfully indexed video {request.video_id}: "
            f"{result['num_chunks']} chunks in {result['duration_seconds']:.2f}s"
        )
        
        return response
    
    except InvalidVideoIDError as e:
        logger.error(f"Invalid video ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except IndexingError as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get("/status/{video_id}", response_model=VideoStatusResponse)
async def get_video_status(video_id: str, namespace: str = ""):
    """
    Check if a video is indexed
    
    Args:
        video_id: YouTube video ID
        namespace: Pinecone namespace
    
    Returns:
        VideoStatusResponse with indexing status
    
    Example:
        GET /index/status/Gfr50f6ZBvo
        
        Response:
        {
            "video_id": "Gfr50f6ZBvo",
            "is_indexed": true,
            "num_chunks": 3
        }
    """
    logger.info(f"Status check for video: {video_id}")
    
    try:
        chain = IndexingChain()
        status_info = chain.check_if_indexed(video_id, namespace=namespace)
        
        response = VideoStatusResponse(**status_info, namespace=namespace)
        
        logger.info(f"Video {video_id} indexed: {response.is_indexed}")
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("", response_model=DeleteResponse)
async def delete_video_index(request: DeleteRequest):
    """
    Delete indexed data for a video
    
    Args:
        request: DeleteRequest with video_id
    
    Returns:
        DeleteResponse with deletion confirmation
    
    Example:
        DELETE /index
        {
            "video_id": "Gfr50f6ZBvo"
        }
    """
    logger.info(f"Delete request for video: {request.video_id}")
    
    try:
        chain = IndexingChain()
        result = chain.delete_video_index(
            video_id=request.video_id,
            namespace=request.namespace
        )
        
        response = DeleteResponse(
            video_id=request.video_id,
            deleted=result.get("deleted", True),
            namespace=request.namespace,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Deleted index for video: {request.video_id}")
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to delete index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )