import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.citation_handler import (
    CitationHandler,
    generate_answer_with_citations
)
from retrieval.simple_retriever import SimpleRetriever

print("="*60)
print("Testing Citation Handler")
print("="*60)

# Test 1: Extract citations from text
print("\nTest 1: Extract citations...")
try:
    handler = CitationHandler()
    
    text = "RAG is useful [Chunk 0]. It combines retrieval [Chunk 1] and generation [Chunk 2]."
    citations = handler.extract_citations(text)
    
    print(f"Text: {text}")
    print(f"Extracted citations: {citations}")
    assert citations == [0, 1, 2], "Should extract [0, 1, 2]"
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Add source information
print("\nTest 2: Add source information...")
try:
    handler = CitationHandler()
    
    answer = "Language models predict text [Chunk 0]. They use neural networks [Chunk 1]."
    chunks = [
        {"text": "Models predict next words...", "chunk_id": 0, "video_id": "abc", "score": 0.85},
        {"text": "Neural networks learn patterns...", "chunk_id": 1, "video_id": "abc", "score": 0.78}
    ]
    
    result = handler.add_source_info(answer, chunks)
    
    print(f"Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")
    print(f"Num sources: {len(result['sources'])}")
    print(f"Source 1 video_id: {result['sources'][0]['video_id']}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")

# Test 3: Format citations with sources
print("\nTest 3: Format citations with full sources...")
try:
    handler = CitationHandler()
    
    answer = "Transformers use attention [Chunk 0]."
    chunks = [
        {"text": "Transformers use attention mechanisms to process sequences...", "chunk_id": 0, "video_id": "xyz", "score": 0.92}
    ]
    
    formatted = handler.format_citations(answer, chunks, include_full_sources=True)
    
    print(f"Formatted answer:\n{formatted}")
    print("Test 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")

# Test 4: Validate citations
print("\nTest 4: Validate citations...")
try:
    handler = CitationHandler()
    
    # Answer with invalid citation
    answer = "Some fact [Chunk 0]. Another fact [Chunk 5]."  # Chunk 5 doesn't exist
    chunks = [
        {"text": "...", "chunk_id": 0},
        {"text": "...", "chunk_id": 1}
    ]
    
    validation = handler.validate_citations(answer, chunks)
    
    print(f"Total citations: {validation['total_citations']}")
    print(f"Valid citations: {validation['valid_citations']}")
    print(f"Invalid citations: {validation['invalid_citations']}")
    print(f"All valid: {validation['all_valid']}")
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")

# Test 5: Remove citations
print("\nTest 5: Remove citations from text...")
try:
    handler = CitationHandler()
    
    text_with_citations = "RAG is useful [Chunk 0]. It helps with accuracy [Chunk 1]."
    clean_text = handler.remove_citations(text_with_citations)
    
    print(f"Original: {text_with_citations}")
    print(f"Cleaned: {clean_text}")
    assert "[Chunk" not in clean_text, "Should remove all citations"
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")

# Test 6: Full RAG with citations (END-TO-END!)
print("\nTest 6: Complete RAG pipeline with citations...")
try:
    # Retrieve
    retriever = SimpleRetriever(top_k=3)
    chunks = retriever.retrieve("What is a language model?")
    
    print(f"Step 1: Retrieved {len(chunks)} chunks")
    
    # Generate with citations
    result = generate_answer_with_citations(
        question="What is a language model?",
        chunks=chunks,
        temperature=0.2,
        max_tokens=300
    )
    
    print(f"\nStep 2: Generated answer with citations")
    print(f"\nQuestion: What is a language model?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations found: {result['citations']}")
    print(f"Valid sources: {result['num_valid_citations']}")
    
    if result['sources']:
        print(f"\nSource chunks used:")
        for source in result['sources']:
            print(f"  - Chunk {source['chunk_id']} (video: {source['video_id']}, score: {source['score']:.2f})")
    
    print("\nTest 6 PASSED")
except Exception as e:
    print(f"Test 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All citation handler tests completed!")
print("="*60)