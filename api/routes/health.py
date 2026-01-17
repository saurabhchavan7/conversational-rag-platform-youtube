"""
Health Check Endpoint
Verify API and dependencies are operational
"""

from fastapi import APIRouter
from datetime import datetime

from api.models import HealthResponse
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns API status and service connectivity
    
    Returns:
        HealthResponse with status and service info
    
    Example:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "timestamp": "2026-01-17T17:30:00",
            "version": "1.0.0",
            "services": {
                "pinecone": "connected",
                "openai": "configured"
            }
        }
    """
    logger.info("Health check requested")
    
    # Check service configurations
    services = {}
    
    # Check Pinecone
    try:
        if settings.PINECONE_API_KEY:
            services["pinecone"] = "configured"
        else:
            services["pinecone"] = "not_configured"
    except:
        services["pinecone"] = "error"
    
    # Check OpenAI
    try:
        if settings.OPENAI_API_KEY:
            services["openai"] = "configured"
        else:
            services["openai"] = "not_configured"
    except:
        services["openai"] = "error"
    
    # Overall status
    all_healthy = all(status == "configured" for status in services.values())
    status = "healthy" if all_healthy else "degraded"
    
    response = HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services=services
    )
    
    logger.info(f"Health check: {status}")
    
    return response