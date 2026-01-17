from utils.exceptions import (
    TranscriptFetchError,
    InvalidVideoIDError,
    LLMError,
    VectorStoreError
)

print("="*60)
print("Testing Custom Exceptions")
print("="*60)

# Test 1: TranscriptFetchError
try:
    raise TranscriptFetchError("Transcript unavailable for video xyz123")
except TranscriptFetchError as e:
    print(f"✅ Caught TranscriptFetchError: {e}")

# Test 2: InvalidVideoIDError
try:
    raise InvalidVideoIDError("Video ID must be 11 characters")
except InvalidVideoIDError as e:
    print(f"✅ Caught InvalidVideoIDError: {e}")

# Test 3: LLMError
try:
    raise LLMError("OpenAI API rate limit exceeded")
except LLMError as e:
    print(f"✅ Caught LLMError: {e}")

# Test 4: VectorStoreError
try:
    raise VectorStoreError("Pinecone connection failed")
except VectorStoreError as e:
    print(f"✅ Caught VectorStoreError: {e}")

print("="*60)
print("All exception tests passed!")
print("="*60)