import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Now imports will work
from retrieval.base_retriever import BaseRetriever

print("="*60)
print("Testing Base Retriever")
print("="*60)

# Test 1: Create a concrete implementation
print("\nTest 1: Create concrete retriever class...")
try:
    class TestRetriever(BaseRetriever):
        """Simple test implementation"""
        
        def retrieve(self, query, top_k=None, filter=None):
            # Mock implementation
            top_k = self._get_top_k(top_k)
            return [
                {"text": f"Result {i}", "score": 0.9 - i*0.1, "video_id": "test", "chunk_id": i}
                for i in range(top_k)
            ]
    
    retriever = TestRetriever(top_k=4)
    print(f"Created retriever: {retriever.__class__.__name__}")
    print(f"Top K: {retriever.top_k}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Test retrieve method
print("\nTest 2: Test retrieve method...")
try:
    retriever = TestRetriever(top_k=4)
    results = retriever.retrieve("test query")
    
    print(f"Retrieved {len(results)} results")
    print(f"First result: {results[0]}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")

# Test 3: Test query validation
print("\nTest 3: Test query validation...")
try:
    retriever = TestRetriever()
    retriever._validate_query("")  # Should raise error
    print("Test 3 FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"Correctly rejected empty query: {e}")
    print("Test 3 PASSED")

# Test 4: Test top_k override
print("\nTest 4: Test top_k override...")
try:
    retriever = TestRetriever(top_k=4)
    results = retriever.retrieve("test", top_k=2)  # Override to 2
    
    print(f"Default top_k: 4")
    print(f"Override top_k: 2")
    print(f"Retrieved: {len(results)} results")
    assert len(results) == 2, "Should return 2 results"
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")

# Test 5: Test retrieve_and_format
print("\nTest 5: Test retrieve_and_format...")
try:
    retriever = TestRetriever(top_k=3)
    response = retriever.retrieve_and_format("test query")
    
    print(f"Query: {response['query']}")
    print(f"Num results: {response['num_results']}")
    print(f"Retriever type: {response['retriever_type']}")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")

print("\n" + "="*60)
print("All base retriever tests completed!")
print("="*60)