from config.settings import settings

print("=" * 60)
print("Configuration loaded successfully!")
print("=" * 60)
print(f"OpenAI Model: {settings.OPENAI_CHAT_MODEL}")
print(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
print(f"Embedding Dimensions: {settings.EMBEDDING_DIMENSIONS}")
print(f"Vector Store: {settings.VECTOR_STORE_TYPE}")
print(f"Pinecone Index: {settings.PINECONE_INDEX_NAME}")
print(f"Pinecone Environment: {settings.PINECONE_ENVIRONMENT}")
print(f"Chunk Size: {settings.CHUNK_SIZE}")
print(f"API Port: {settings.API_PORT}")
print("=" * 60)