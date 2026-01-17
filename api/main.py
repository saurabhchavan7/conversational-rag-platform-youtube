"""
FastAPI Main Application
Entry point for the Conversational RAG Platform API
"""

import sys
from pathlib import Path

# Add project root to path (for direct execution)
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, index, query
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Conversational RAG Platform API",
    description="Production-grade RAG system for YouTube content interaction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(index.router)
app.include_router(query.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for uncaught errors
    """
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Execute on application startup
    """
    logger.info("="*60)
    logger.info("Starting Conversational RAG Platform API")
    logger.info("="*60)
    logger.info(f"OpenAI Model: {settings.OPENAI_CHAT_MODEL}")
    logger.info(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
    logger.info(f"Vector Store: {settings.VECTOR_STORE_TYPE}")
    logger.info(f"Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    logger.info(f"Embedding Dimensions: {settings.EMBEDDING_DIMENSIONS}")
    logger.info(f"API Port: {settings.API_PORT}")
    logger.info("="*60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Execute on application shutdown
    """
    logger.info("Shutting down Conversational RAG Platform API")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Conversational RAG Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "index_video": "POST /index",
            "query_video": "POST /query",
            "video_status": "GET /index/status/{video_id}",
            "delete_video": "DELETE /index"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )