"""
API Models - Request and Response Schemas
Pydantic models for API validation and documentation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================
# Request Models
# ============================================

class IndexRequest(BaseModel):
    """Request model for indexing a video"""
    
    video_id: str = Field(
        ...,
        description="YouTube video ID (11 characters)",
        min_length=11,
        max_length=11,
        example="Gfr50f6ZBvo"
    )
    
    namespace: Optional[str] = Field(
        default="",
        description="Pinecone namespace for organization",
        example=""
    )
    
    chunk_size: Optional[int] = Field(
        default=None,
        description="Override default chunk size",
        ge=100,
        le=5000,
        example=1000
    )
    
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="Override default chunk overlap",
        ge=0,
        le=1000,
        example=200
    )
    
    @validator('video_id')
    def validate_video_id_format(cls, v):
        """Validate video ID contains only valid characters"""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', v):
            raise ValueError('Invalid video ID format')
        return v


class QueryRequest(BaseModel):
    """Request model for querying the system"""
    
    question: str = Field(
        ...,
        description="User's question",
        min_length=3,
        max_length=500,
        example="What is a language model?"
    )
    
    video_id: Optional[str] = Field(
        default=None,
        description="Filter to specific video (optional)",
        min_length=11,
        max_length=11,
        example="Gfr50f6ZBvo"
    )
    
    retriever_type: str = Field(
        default="hybrid",
        description="Retrieval strategy to use",
        example="hybrid"
    )
    
    top_k: Optional[int] = Field(
        default=4,
        description="Number of chunks to retrieve",
        ge=1,
        le=10,
        example=4
    )
    
    include_citations: bool = Field(
        default=True,
        description="Include source citations in answer",
        example=True
    )
    
    temperature: Optional[float] = Field(
        default=0.2,
        description="LLM temperature for generation",
        ge=0.0,
        le=1.0,
        example=0.2
    )
    
    @validator('retriever_type')
    def validate_retriever_type(cls, v):
        """Validate retriever type is supported"""
        valid_types = ['simple', 'rewriting', 'hybrid']
        if v not in valid_types:
            raise ValueError(f'retriever_type must be one of {valid_types}')
        return v
    
    @validator('video_id')
    def validate_video_id_format(cls, v):
        """Validate video ID format if provided"""
        if v is None:
            return v
        
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', v):
            raise ValueError('Invalid video ID format')
        return v


# ============================================
# Response Models
# ============================================

class HealthResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2026-01-17T17:30:00")
    version: str = Field(..., example="1.0.0")
    services: Dict[str, str] = Field(
        ...,
        example={
            "pinecone": "connected",
            "openai": "configured"
        }
    )


class IndexResponse(BaseModel):
    """Response model for indexing operation"""
    
    video_id: str = Field(..., example="Gfr50f6ZBvo")
    status: str = Field(..., example="success")
    num_chunks: int = Field(..., example=3)
    num_embeddings: int = Field(..., example=3)
    num_stored: int = Field(..., example=3)
    transcript_chars: int = Field(..., example=2118)
    duration_seconds: float = Field(..., example=6.45)
    namespace: str = Field(..., example="")
    timestamp: str = Field(..., example="2026-01-17T17:30:00")


class SourceInfo(BaseModel):
    """Model for source chunk information"""
    
    chunk_id: int = Field(..., example=0)
    text: str = Field(..., example="Language models are...")
    video_id: str = Field(..., example="Gfr50f6ZBvo")
    score: float = Field(..., example=0.85)


class QueryResponse(BaseModel):
    """Response model for query operation"""
    
    question: str = Field(..., example="What is a language model?")
    answer: str = Field(..., example="A language model is...")
    citations: List[int] = Field(
        default_factory=list,
        example=[0, 1, 2]
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        example=[]
    )
    num_sources: int = Field(default=0, example=2)
    retrieved_chunks: int = Field(..., example=4)
    retriever_type: str = Field(..., example="hybrid")
    duration_seconds: float = Field(..., example=2.89)
    timestamp: str = Field(..., example="2026-01-17T17:30:00")


class VideoStatusResponse(BaseModel):
    """Response model for video status check"""
    
    video_id: str = Field(..., example="Gfr50f6ZBvo")
    is_indexed: bool = Field(..., example=True)
    num_chunks: Optional[int] = Field(default=0, example=3)
    namespace: str = Field(default="", example="")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    
    error: str = Field(..., example="Invalid video ID format")
    detail: Optional[str] = Field(
        default=None,
        example="Video ID must be exactly 11 characters"
    )
    timestamp: str = Field(..., example="2026-01-17T17:30:00")


# ============================================
# Utility Models
# ============================================

class BatchIndexRequest(BaseModel):
    """Request model for batch indexing (future feature)"""
    
    video_ids: List[str] = Field(
        ...,
        description="List of video IDs to index",
        min_items=1,
        max_items=10,
        example=["Gfr50f6ZBvo", "dQw4w9WgXcQ"]
    )
    
    namespace: Optional[str] = Field(default="", example="")


class DeleteRequest(BaseModel):
    """Request model for deleting video index"""
    
    video_id: str = Field(
        ...,
        description="YouTube video ID to delete",
        min_length=11,
        max_length=11,
        example="Gfr50f6ZBvo"
    )
    
    namespace: Optional[str] = Field(default="", example="")


class DeleteResponse(BaseModel):
    """Response model for deletion operation"""
    
    video_id: str = Field(..., example="Gfr50f6ZBvo")
    deleted: bool = Field(..., example=True)
    namespace: str = Field(..., example="")
    timestamp: str = Field(..., example="2026-01-17T17:30:00")