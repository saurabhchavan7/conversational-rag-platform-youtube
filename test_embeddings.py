from indexing.embeddings import EmbeddingGenerator, generate_embeddings_for_chunks
from indexing.text_splitter import split_transcript_into_chunks
from indexing.document_loader import load_youtube_transcript

print("="*60)
print("Testing Embedding Generator")
print("="*60)

# Test 1: Generate single embedding
print("\nTest 1: Generate single embedding...")
try:
    generator = EmbeddingGenerator()
    
    text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(text)
    
    print(f"✅ Generated embedding with {len(embedding)} dimensions")
    print(f"✅ First 5 values: {embedding[:5]}")
    print(f"✅ Expected dimensions: {generator.dimensions}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"❌ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Batch embedding generation
print("\nTest 2: Batch embedding generation...")
try:
    generator = EmbeddingGenerator()
    
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    embeddings = generator.generate_embeddings_batch(texts)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"✅ Each embedding has {len(embeddings[0])} dimensions")
    print("Test 2 PASSED")
except Exception as e:
    print(f"❌ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Embed chunks from transcript
print("\nTest 3: Embedding transcript chunks...")
try:
    # Load and split transcript
    transcript = load_youtube_transcript("Gfr50f6ZBvo")
    chunks = split_transcript_into_chunks(transcript, chunk_size=300, chunk_overlap=50)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embedded_chunks = generator.embed_chunks(chunks, show_progress=True)
    
    print(f"✅ Embedded {len(embedded_chunks)} chunks")
    print(f"✅ First chunk has embedding: {'embedding' in embedded_chunks[0]}")
    print(f"✅ Embedding dimensions: {embedded_chunks[0]['embedding_dim']}")
    print("Test 3 PASSED")
except Exception as e:
    print(f"❌ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Convenience function
print("\nTest 4: Using convenience function...")
try:
    transcript = load_youtube_transcript("Gfr50f6ZBvo")
    chunks = split_transcript_into_chunks(transcript, chunk_size=400)
    
    embedded_chunks = generate_embeddings_for_chunks(chunks)
    
    print(f"✅ Embedded {len(embedded_chunks)} chunks using convenience function")
    print("Test 4 PASSED")
except Exception as e:
    print(f"❌ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All embedding tests completed!")
print("="*60)