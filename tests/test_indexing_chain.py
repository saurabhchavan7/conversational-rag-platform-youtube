import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chains.indexing_chain import IndexingChain, index_youtube_video

print("="*60)
print("Testing Indexing Chain")
print("="*60)

# Test 1: Initialize chain
print("\nTest 1: Initialize IndexingChain...")
try:
    chain = IndexingChain()
    
    print(f"Initialized: {chain.__class__.__name__}")
    print(f"Chunk size: {chain.chunk_size}")
    print(f"Chunk overlap: {chain.chunk_overlap}")
    print(f"Embedding dimensions: {chain.embedding_dimensions}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Check if video is indexed
print("\nTest 2: Check if video is indexed...")
try:
    chain = IndexingChain()
    
    status = chain.check_if_indexed("Gfr50f6ZBvo")
    
    print(f"Video ID: {status['video_id']}")
    print(f"Is indexed: {status['is_indexed']}")
    if status['is_indexed']:
        print(f"Num chunks: {status.get('num_chunks', 0)}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Index a new video (COMPLETE PIPELINE!)
print("\nTest 3: Index complete video (full pipeline)...")
try:
    chain = IndexingChain()
    
    print("Starting full indexing pipeline...")
    print("This will take ~10 seconds...")
    
    result = chain.index_video("dQw4w9WgXcQ")
    
    print(f"\nIndexing Results:")
    print(f"  Video ID: {result['video_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Chunks created: {result['num_chunks']}")
    print(f"  Embeddings generated: {result['num_embeddings']}")
    print(f"  Vectors stored: {result['num_stored']}")
    print(f"  Transcript size: {result['transcript_chars']} characters")
    print(f"  Duration: {result['duration_seconds']:.2f} seconds")
    print(f"  Namespace: '{result['namespace']}'")
    
    print("\nTest 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Convenience function
print("\nTest 4: Using convenience function...")
try:
    result = index_youtube_video("Gfr50f6ZBvo", chunk_size=500)
    
    print(f"Indexed video: {result['video_id']}")
    print(f"Chunks: {result['num_chunks']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print("Test 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Delete video index
print("\nTest 5: Delete video index...")
try:
    chain = IndexingChain()
    
    # Delete a test video
    result = chain.delete_video_index("dQw4w9WgXcQ")
    
    print(f"Deleted: {result.get('deleted', False)}")
    print(f"Video ID: {result.get('video_id', 'unknown')}")
    print("Test 5 PASSED")
except Exception as e:
    print(f"Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All indexing chain tests completed!")
print("="*60)