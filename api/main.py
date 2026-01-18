"""
FastAPI Main Application
Entry point for the RAG Platform API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, index, query
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YouTube RAG Platform API",
    description="RAG system for conversational interaction with YouTube videos",
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


@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("="*60)
    logger.info("Starting YouTube RAG Platform API")
    logger.info("="*60)
    logger.info(f"OpenAI Model: {settings.OPENAI_CHAT_MODEL}")
    logger.info(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
    logger.info(f"Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    logger.info(f"API Port: {settings.API_PORT}")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("Shutting down YouTube RAG Platform API")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YouTube RAG Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "index_video": "POST /index",
            "query_video": "POST /query",
            "video_status": "GET /index/status/{video_id}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )