from indexing.text_splitter import (
    TranscriptTextSplitter,
    split_transcript_into_chunks
)
from indexing.document_loader import load_youtube_transcript

print("="*60)
print("Testing Text Splitter")
print("="*60)

# Test 1: Basic text splitting
print("\nTest 1: Basic text splitting...")
try:
    splitter = TranscriptTextSplitter(chunk_size=500, chunk_overlap=100)
    
    text = "This is a test sentence. " * 50  # ~1250 characters
    chunks = splitter.split_text(text)
    
    print(f"✅ Original text length: {len(text)} chars")
    print(f"✅ Number of chunks: {len(chunks)}")
    print(f"✅ First chunk preview: {chunks[0]['text'][:80]}...")
    print(f"✅ Chunk 0 length: {chunks[0]['chunk_length']} chars")
    print("Test 1 PASSED")
except Exception as e:
    print(f"❌ Test 1 FAILED: {e}")

# Test 2: Split real transcript (mock data)
print("\nTest 2: Splitting mock YouTube transcript...")
try:
    # Load transcript (mock)
    transcript = load_youtube_transcript("Gfr50f6ZBvo")
    print(f"✅ Loaded transcript: {transcript['total_chars']} chars")
    
    # Split transcript
    splitter = TranscriptTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_transcript(transcript)
    
    print(f"✅ Split into {len(chunks)} chunks")
    print(f"✅ First chunk preview: {chunks[0]['text'][:100]}...")
    print(f"✅ Chunk metadata: {chunks[0]['metadata']}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"❌ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Get chunk statistics
print("\nTest 3: Chunk statistics...")
try:
    splitter = TranscriptTextSplitter()
    transcript = load_youtube_transcript("Gfr50f6ZBvo")
    chunks = splitter.split_transcript(transcript)
    
    stats = splitter.get_chunk_stats(chunks)
    
    print(f"✅ Total chunks: {stats['total_chunks']}")
    print(f"✅ Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    print(f"✅ Min chunk length: {stats['min_chunk_length']} chars")
    print(f"✅ Max chunk length: {stats['max_chunk_length']} chars")
    print("Test 3 PASSED")
except Exception as e:
    print(f"❌ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Convenience function
print("\nTest 4: Using convenience function...")
try:
    transcript = load_youtube_transcript("Gfr50f6ZBvo")
    chunks = split_transcript_into_chunks(transcript, chunk_size=400, chunk_overlap=80)
    
    print(f"✅ Created {len(chunks)} chunks with custom settings")
    print("Test 4 PASSED")
except Exception as e:
    print(f"❌ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All text splitter tests completed!")
print("="*60)