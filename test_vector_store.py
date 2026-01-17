from indexing.vector_store import PineconeVectorStore, store_embeddings_in_pinecone
from indexing.embeddings import generate_embeddings_for_chunks
from indexing.text_splitter import split_transcript_into_chunks
from indexing.document_loader import load_youtube_transcript

print("="*60)
print("Testing Pinecone Vector Store")
print("="*60)

# Test 1: Initialize vector store
print("\nTest 1: Initialize Pinecone connection...")
try:
    store = PineconeVectorStore()
    print(f"✅ Connected to index: {store.index_name}")
    print(f"✅ Dimensions: {store.dimension}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"❌ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Get index stats
print("\nTest 2: Get index statistics...")
try:
    store = PineconeVectorStore()
    stats = store.get_index_stats()
    
    print(f"✅ Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"✅ Dimension: {stats.get('dimension', 0)}")
    print(f"✅ Index fullness: {stats.get('index_fullness', 0)}")
    print("Test 2 PASSED")
except Exception as e:
    print(f"❌ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Upsert chunks to Pinecone
print("\nTest 3: Upsert chunks to Pinecone...")
embedded_chunks = None  # Initialize to avoid NameError
try:
    # Prepare data (use valid 11-character video ID)
    transcript = load_youtube_transcript("dQw4w9WgXcQ")  # Valid 11-char ID
    chunks = split_transcript_into_chunks(transcript, chunk_size=300, chunk_overlap=50)
    embedded_chunks = generate_embeddings_for_chunks(chunks)
    
    print(f"✅ Prepared {len(embedded_chunks)} embedded chunks")
    
    # Upsert to Pinecone
    store = PineconeVectorStore()
    result = store.upsert_chunks(embedded_chunks)
    
    print(f"✅ Upserted {result['upserted_count']} vectors")
    print(f"✅ Namespace: '{result['namespace']}'")
    print("Test 3 PASSED")
except Exception as e:
    print(f"❌ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Query Pinecone
print("\nTest 4: Query similar vectors...")
try:
    if embedded_chunks is None:
        print("⚠️ Skipping Test 4: No embedded chunks from Test 3")
    else:
        store = PineconeVectorStore()
        
        # Use the first chunk's embedding as query
        query_embedding = embedded_chunks[0]["embedding"]
        
        results = store.query(query_embedding, top_k=3)
        
        print(f"✅ Found {len(results)} matches")
        for i, match in enumerate(results):
            print(f"  Match {i+1}: score={match['score']:.4f}, text={match['text'][:60]}...")
        print("Test 4 PASSED")
except Exception as e:
    print(f"❌ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Convenience function
print("\nTest 5: Using convenience function...")
try:
    transcript = load_youtube_transcript("Gfr50f6ZBvo")  # Valid 11-char ID
    chunks = split_transcript_into_chunks(transcript)
    embedded_chunks = generate_embeddings_for_chunks(chunks)
    
    result = store_embeddings_in_pinecone(embedded_chunks, video_id="Gfr50f6ZBvo")
    
    print(f"✅ Stored {result['upserted_count']} vectors using convenience function")
    print("Test 5 PASSED")
except Exception as e:
    print(f"❌ Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All vector store tests completed!")
print("="*60)