"""
Run the FastAPI application
"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        reload_excludes=["logs/*", "*.log", "__pycache__/*", "*.pyc"],
        log_level=settings.LOG_LEVEL.lower()
    )