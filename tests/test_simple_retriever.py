import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.simple_retriever import SimpleRetriever, search_pinecone

print("="*60)
print("Testing Simple Retriever")
print("="*60)

# Test 1: Initialize retriever
print("\nTest 1: Initialize SimpleRetriever...")
try:
    retriever = SimpleRetriever(top_k=4)
    print(f"Initialized: {retriever.__class__.__name__}")
    print(f"Top K: {retriever.top_k}")
    print(f"Vector store: {retriever.vector_store.index_name}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Retrieve with a query
print("\nTest 2: Retrieve results for query...")
try:
    retriever = SimpleRetriever(top_k=3)
    results = retriever.retrieve("What is a language model?")
    
    print(f"Query: 'What is a language model?'")
    print(f"Retrieved: {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Video ID: {result['video_id']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Text preview: {result['text'][:100]}...")
    
    print("\nTest 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Retrieve from specific video
print("\nTest 3: Retrieve from specific video...")
try:
    retriever = SimpleRetriever()
    results = retriever.retrieve_for_video(
        query="transformer model",
        video_id="Gfr50f6ZBvo",
        top_k=2
    )
    
    print(f"Retrieved {len(results)} results from video Gfr50f6ZBvo")
    for result in results:
        print(f"  - Video: {result['video_id']}, Score: {result['score']:.4f}")
    
    print("Test 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Convenience function
print("\nTest 4: Using convenience function...")
try:
    results = search_pinecone("machine learning", top_k=2)
    
    print(f"Retrieved {len(results)} results using convenience function")
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All simple retriever tests completed!")
print("="*60)