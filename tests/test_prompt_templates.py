import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from augmentation.prompt_templates import (
    format_context_for_prompt,
    build_qa_prompt,
    build_summary_prompt,
    check_prompt_length,
    QA_SYSTEM_PROMPT
)

print("="*60)
print("Testing Prompt Templates")
print("="*60)

# Test 1: Format context without chunk IDs
print("\nTest 1: Format context without chunk IDs...")
try:
    chunks = [
        {"text": "First chunk about AI", "chunk_id": 0},
        {"text": "Second chunk about ML", "chunk_id": 1},
        {"text": "Third chunk about NLP", "chunk_id": 2}
    ]
    
    context = format_context_for_prompt(chunks, include_chunk_ids=False)
    
    print(f"Formatted {len(chunks)} chunks")
    print(f"Context length: {len(context)} characters")
    print(f"Context preview:\n{context[:150]}...")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Format context WITH chunk IDs (for citations)
print("\nTest 2: Format context with chunk IDs...")
try:
    context = format_context_for_prompt(chunks, include_chunk_ids=True)
    
    print(f"Context with IDs:\n{context[:200]}...")
    print(f"Contains '[Chunk 0]': {'[Chunk 0]' in context}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")

# Test 3: Build QA prompt without citations
print("\nTest 3: Build QA prompt without citations...")
try:
    question = "What is machine learning?"
    prompt = build_qa_prompt(question, chunks, include_citations=False)
    
    print(f"Question: {question}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Contains question: {question in prompt}")
    print(f"Contains context: {'First chunk about AI' in prompt}")
    print("Test 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")

# Test 4: Build QA prompt WITH citations
print("\nTest 4: Build QA prompt with citations...")
try:
    prompt = build_qa_prompt(question, chunks, include_citations=True)
    
    print(f"Prompt requests citations: {'citation' in prompt.lower()}")
    print(f"Contains chunk IDs: {'[Chunk' in prompt}")
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")

# Test 5: Build summary prompt
print("\nTest 5: Build summary prompt...")
try:
    summary_prompt = build_summary_prompt(chunks)
    
    print(f"Summary prompt length: {len(summary_prompt)} characters")
    print(f"Contains 'summary': {'summary' in summary_prompt.lower()}")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")

# Test 6: Check prompt length
print("\nTest 6: Check prompt length...")
try:
    test_prompt = "This is a test prompt. " * 500  # ~11,500 chars
    
    check = check_prompt_length(test_prompt, max_tokens=3000)
    
    print(f"Estimated tokens: {check['estimated_tokens']}")
    print(f"Max tokens: {check['max_tokens']}")
    print(f"Exceeds limit: {check['exceeds_limit']}")
    print(f"Usage: {check['usage_pct']:.1f}%")
    print("Test 6 PASSED")
except Exception as e:
    print(f"Test 6 FAILED: {e}")

# Test 7: Test with real retrieval results
print("\nTest 7: Build prompt from real retrieval...")
try:
    from retrieval.simple_retriever import SimpleRetriever
    
    # Retrieve chunks
    retriever = SimpleRetriever(top_k=3)
    results = retriever.retrieve("What is a language model?")
    
    # Build prompt
    prompt = build_qa_prompt(
        question="What is a language model?",
        chunks=results,
        include_citations=True
    )
    
    print(f"Built prompt from {len(results)} retrieved chunks")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Check length
    check = check_prompt_length(prompt)
    print(f"Token estimate: {check['estimated_tokens']} tokens")
    print(f"Within limits: {not check['exceeds_limit']}")
    
    print("Test 7 PASSED")
except Exception as e:
    print(f"Test 7 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All prompt template tests completed!")
print("="*60)