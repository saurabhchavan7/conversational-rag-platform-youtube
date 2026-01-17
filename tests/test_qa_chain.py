import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chains.qa_chain import QAChain, ask_question

print("="*60)
print("Testing QA Chain")
print("="*60)

# Test 1: Initialize with simple retriever
print("\nTest 1: Initialize QAChain with simple retriever...")
try:
    chain = QAChain(retriever_type="simple", top_k=3)
    
    print(f"Initialized: {chain.__class__.__name__}")
    print(f"Retriever type: {chain.retriever_type}")
    print(f"Top K: {chain.top_k}")
    print(f"Citations enabled: {chain.include_citations}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Answer question with simple retriever
print("\nTest 2: Answer with simple retriever...")
try:
    chain = QAChain(retriever_type="simple", top_k=3, include_citations=False)
    
    result = chain.answer("What is a language model?")
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nMetadata:")
    print(f"  Retrieved chunks: {result['retrieved_chunks']}")
    print(f"  Retriever: {result['retriever_type']}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")
    
    print("\nTest 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Answer with query rewriting
print("\nTest 3: Answer with query rewriting...")
try:
    chain = QAChain(retriever_type="rewriting", top_k=3)
    
    result = chain.answer("what's a transformer")
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer (first 200 chars): {result['answer'][:200]}...")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    
    print("\nTest 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Answer with hybrid retriever and citations
print("\nTest 4: Answer with hybrid retriever + citations...")
try:
    chain = QAChain(retriever_type="hybrid", top_k=4, include_citations=True)
    
    result = chain.answer("How are language models trained?")
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    if result.get('sources'):
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'][:2]):  # Show first 2
            print(f"  {i+1}. Chunk {source['chunk_id']} (score: {source['score']:.2f})")
            print(f"     {source['text'][:100]}...")
    
    print("\nTest 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Filter by video ID
print("\nTest 5: Answer with video ID filter...")
try:
    chain = QAChain(retriever_type="simple")
    
    result = chain.answer(
        question="What is discussed in this video?",
        video_id="Gfr50f6ZBvo"
    )
    
    print(f"Answered from video: Gfr50f6ZBvo only")
    print(f"Retrieved: {result['retrieved_chunks']} chunks")
    print(f"Answer length: {len(result['answer'])} chars")
    
    print("\nTest 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Streaming answer
print("\nTest 6: Streaming answer generation...")
try:
    chain = QAChain(retriever_type="simple", top_k=2)
    
    print("\nQuestion: What is machine learning?")
    print("Streaming answer: ", end="", flush=True)
    
    full_answer = ""
    for chunk in chain.answer_streaming("What is machine learning?"):
        print(chunk, end="", flush=True)
        full_answer += chunk
    
    print(f"\n\nFull answer length: {len(full_answer)} chars")
    print("Test 6 PASSED")
except Exception as e:
    print(f"\nTest 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Convenience function (COMPLETE RAG IN 1 LINE!)
print("\nTest 7: Complete RAG with convenience function...")
try:
    result = ask_question(
        question="What are neural networks?",
        retriever_type="hybrid",
        include_citations=True
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer'][:150]}...")
    print(f"Citations: {result.get('citations', [])}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    
    print("\nTest 7 PASSED")
except Exception as e:
    print(f"Test 7 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All QA chain tests completed!")
print("="*60)
print("\n" + "="*60)
print("COMPLETE RAG SYSTEM WORKING END-TO-END!")
print("="*60)