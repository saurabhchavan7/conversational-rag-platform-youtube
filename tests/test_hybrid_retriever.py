import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_retriever import HybridRetriever, hybrid_search

print("="*60)
print("Testing Hybrid Retriever")
print("="*60)

# Test 1: Initialize hybrid retriever
print("\nTest 1: Initialize HybridRetriever...")
try:
    retriever = HybridRetriever(
        top_k=4,
        dense_weight=0.7,
        sparse_weight=0.3
    )
    print(f"Initialized: {retriever.__class__.__name__}")
    print(f"Dense weight: {retriever.dense_weight}")
    print(f"Sparse weight: {retriever.sparse_weight}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Hybrid retrieval
print("\nTest 2: Hybrid retrieval...")
try:
    retriever = HybridRetriever(top_k=3)
    
    # Query with specific keyword that might not be semantically close
    results = retriever.retrieve("language model parameters")
    
    print(f"Query: 'language model parameters'")
    print(f"Retrieved: {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Hybrid Score: {result['score']:.4f}")
        print(f"  Dense Score: {result.get('dense_score', 0):.4f}")
        print(f"  Sparse Score: {result.get('sparse_score', 0):.4f}")
        print(f"  Text: {result['text'][:80]}...")
    
    print("\nTest 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare with simple retriever
print("\nTest 3: Compare hybrid vs simple retrieval...")
try:
    from retrieval.simple_retriever import SimpleRetriever
    
    query = "transformer model"
    
    # Simple retrieval
    simple = SimpleRetriever(top_k=3)
    simple_results = simple.retrieve(query)
    
    # Hybrid retrieval
    hybrid = HybridRetriever(top_k=3)
    hybrid_results = hybrid.retrieve(query)
    
    print(f"Query: '{query}'")
    print(f"\nSimple retriever:")
    print(f"  Avg score: {sum(r['score'] for r in simple_results)/len(simple_results):.4f}")
    
    print(f"\nHybrid retriever:")
    print(f"  Avg score: {sum(r['score'] for r in hybrid_results)/len(hybrid_results):.4f}")
    
    print("\nTest 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with different weights
print("\nTest 4: Test different weight configurations...")
try:
    # More semantic focus
    retriever_semantic = HybridRetriever(top_k=2, dense_weight=0.9, sparse_weight=0.1)
    results_semantic = retriever_semantic.retrieve("machine learning")
    
    print(f"Semantic-focused (0.9/0.1): {len(results_semantic)} results")
    
    # More keyword focus
    retriever_keyword = HybridRetriever(top_k=2, dense_weight=0.3, sparse_weight=0.7)
    results_keyword = retriever_keyword.retrieve("machine learning")
    
    print(f"Keyword-focused (0.3/0.7): {len(results_keyword)} results")
    
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Convenience function
print("\nTest 5: Using convenience function...")
try:
    results = hybrid_search("neural network training", top_k=3)
    
    print(f"Retrieved {len(results)} results using convenience function")
    print(f"First result hybrid score: {results[0]['score']:.4f}")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All hybrid retriever tests completed!")
print("="*60)