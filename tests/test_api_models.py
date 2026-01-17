import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    IndexRequest,
    QueryRequest,
    IndexResponse,
    QueryResponse,
    VideoStatusResponse,
    ErrorResponse
)
from pydantic import ValidationError

print("="*60)
print("Testing API Models")
print("="*60)

# Test 1: Valid IndexRequest
print("\nTest 1: Valid IndexRequest...")
try:
    request = IndexRequest(
        video_id="Gfr50f6ZBvo",
        namespace="",
        chunk_size=1000
    )
    
    print(f"Video ID: {request.video_id}")
    print(f"Chunk size: {request.chunk_size}")
    print(f"Valid request created")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Invalid IndexRequest (wrong video ID length)
print("\nTest 2: Invalid IndexRequest (wrong length)...")
try:
    request = IndexRequest(video_id="short")
    print("Test 2 FAILED: Should have raised ValidationError")
except ValidationError as e:
    print("Correctly rejected invalid video ID")
    print("Test 2 PASSED")

# Test 3: Valid QueryRequest
print("\nTest 3: Valid QueryRequest...")
try:
    request = QueryRequest(
        question="What is RAG?",
        video_id="Gfr50f6ZBvo",
        retriever_type="hybrid",
        top_k=4
    )
    
    print(f"Question: {request.question}")
    print(f"Retriever: {request.retriever_type}")
    print(f"Top K: {request.top_k}")
    print("Test 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")

# Test 4: Invalid retriever type
print("\nTest 4: Invalid retriever type...")
try:
    request = QueryRequest(
        question="Test",
        retriever_type="invalid_type"
    )
    print("Test 4 FAILED: Should have raised ValidationError")
except ValidationError as e:
    print("Correctly rejected invalid retriever type")
    print("Test 4 PASSED")

# Test 5: IndexResponse
print("\nTest 5: IndexResponse model...")
try:
    response = IndexResponse(
        video_id="Gfr50f6ZBvo",
        status="success",
        num_chunks=3,
        num_embeddings=3,
        num_stored=3,
        transcript_chars=2118,
        duration_seconds=6.45,
        namespace="",
        timestamp="2026-01-17T17:30:00"
    )
    
    print(f"Response created: {response.status}")
    print(f"Chunks: {response.num_chunks}")
    print(f"Duration: {response.duration_seconds}s")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")

# Test 6: QueryResponse
print("\nTest 6: QueryResponse model...")
try:
    response = QueryResponse(
        question="What is ML?",
        answer="Machine learning is...",
        citations=[0, 1],
        sources=[],
        num_sources=2,
        retrieved_chunks=4,
        retriever_type="hybrid",
        duration_seconds=2.89,
        timestamp="2026-01-17T17:30:00"
    )
    
    print(f"Question: {response.question}")
    print(f"Answer length: {len(response.answer)} chars")
    print(f"Citations: {response.citations}")
    print("Test 6 PASSED")
except Exception as e:
    print(f"Test 6 FAILED: {e}")

# Test 7: ErrorResponse
print("\nTest 7: ErrorResponse model...")
try:
    response = ErrorResponse(
        error="Invalid video ID",
        detail="Video ID must be 11 characters",
        timestamp="2026-01-17T17:30:00"
    )
    
    print(f"Error: {response.error}")
    print(f"Detail: {response.detail}")
    print("Test 7 PASSED")
except Exception as e:
    print(f"Test 7 FAILED: {e}")

print("\n" + "="*60)
print("All API model tests completed!")
print("="*60)