import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.query_rewriter import (
    QueryRewritingRetriever,
    retrieve_with_rewriting
)

print("="*60)
print("Testing Query Rewriting Retriever")
print("="*60)

# Test 1: Initialize retriever
print("\nTest 1: Initialize QueryRewritingRetriever...")
try:
    retriever = QueryRewritingRetriever(top_k=3)
    print(f"Initialized: {retriever.__class__.__name__}")
    print(f"LLM Model: {retriever.llm_model}")
    print(f"Temperature: {retriever.temperature}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Rewrite a vague query
print("\nTest 2: Rewrite vague query...")
try:
    retriever = QueryRewritingRetriever()
    
    original = "what's rag"
    rewritten = retriever.rewrite_query(original)
    
    print(f"Original query: '{original}'")
    print(f"Rewritten query: '{rewritten}'")
    print(f"Query improved: {len(rewritten) > len(original)}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Retrieve with query rewriting
print("\nTest 3: Retrieve with query rewriting...")
try:
    retriever = QueryRewritingRetriever(top_k=3)
    
    results = retriever.retrieve("what's a language model")
    
    print(f"Retrieved: {len(results)} results")
    print(f"Original query: {results[0].get('original_query', 'N/A')}")
    print(f"Rewritten query: {results[0].get('rewritten_query', 'N/A')}")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['text'][:80]}...")
    
    print("\nTest 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Compare with and without rewriting
print("\nTest 4: Compare original vs rewritten query...")
try:
    retriever = QueryRewritingRetriever()
    
    comparison = retriever.compare_queries("transformer", top_k=2)
    
    print(f"Original query: '{comparison['original_query']}'")
    print(f"Rewritten query: '{comparison['rewritten_query']}'")
    print(f"\nOriginal avg score: {comparison['improvement']['original_avg_score']:.4f}")
    print(f"Rewritten avg score: {comparison['improvement']['rewritten_avg_score']:.4f}")
    print(f"Improvement: {comparison['improvement']['improvement_pct']:.2f}%")
    
    print("\nTest 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Convenience function
print("\nTest 5: Using convenience function...")
try:
    results = retrieve_with_rewriting("llm training", top_k=2)
    
    print(f"Retrieved {len(results)} results")
    print(f"Rewritten query: {results[0]['rewritten_query']}")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All query rewriter tests completed!")
print("="*60)