import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.llm_client import LLMClient, generate_answer
from augmentation.prompt_templates import build_qa_prompt
from retrieval.simple_retriever import SimpleRetriever

print("="*60)
print("Testing LLM Client")
print("="*60)

# Test 1: Initialize LLM client
print("\nTest 1: Initialize LLM client...")
try:
    client = LLMClient()
    
    print(f"Initialized: {client.__class__.__name__}")
    print(f"Model: {client.model}")
    print(f"Temperature: {client.temperature}")
    print(f"Max tokens: {client.max_tokens}")
    print("Test 1 PASSED")
except Exception as e:
    print(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Generate simple answer
print("\nTest 2: Generate simple answer...")
try:
    client = LLMClient(temperature=0.0, max_tokens=100)
    
    prompt = """You are a helpful assistant.

Question: What is 2+2?

Answer:"""
    
    answer = client.generate(prompt)
    
    print(f"Prompt: 'What is 2+2?'")
    print(f"Answer: '{answer}'")
    print(f"Answer length: {len(answer)} characters")
    print("Test 2 PASSED")
except Exception as e:
    print(f"Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Generate with system prompt
print("\nTest 3: Generate with system prompt...")
try:
    client = LLMClient()
    
    answer = client.generate_with_system_prompt(
        system_prompt="You are a helpful AI assistant. Be concise.",
        user_message="Explain what an API is in one sentence."
    )
    
    print(f"Generated answer: {answer}")
    print("Test 3 PASSED")
except Exception as e:
    print(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Real RAG pipeline (Retrieve + Generate)
print("\nTest 4: Full RAG pipeline (Retrieve -> Generate)...")
try:
    # Step 1: Retrieve relevant chunks
    retriever = SimpleRetriever(top_k=3)
    chunks = retriever.retrieve("What is a language model?")
    
    print(f"Step 1: Retrieved {len(chunks)} chunks")
    
    # Step 2: Build prompt
    prompt = build_qa_prompt(
        question="What is a language model?",
        chunks=chunks,
        include_citations=False
    )
    
    print(f"Step 2: Built prompt ({len(prompt)} chars)")
    
    # Step 3: Generate answer
    client = LLMClient(temperature=0.2, max_tokens=300)
    answer = client.generate(prompt)
    
    print(f"Step 3: Generated answer")
    print(f"\nQuestion: What is a language model?")
    print(f"\nAnswer:\n{answer}")
    
    print("\nTest 4 PASSED")
except Exception as e:
    print(f"Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Streaming generation
print("\nTest 5: Test streaming generation...")
try:
    client = LLMClient(max_tokens=100)
    
    prompt = "Explain AI in 2 sentences."
    
    print("Streaming response: ", end="", flush=True)
    full_response = ""
    for chunk in client.generate_streaming(prompt):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print(f"\n\nFull response length: {len(full_response)} chars")
    print("Test 5 PASSED")
except Exception as e:
    print(f"\nTest 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Convenience function
print("\nTest 6: Using convenience function...")
try:
    retriever = SimpleRetriever(top_k=2)
    chunks = retriever.retrieve("training models")
    
    prompt = build_qa_prompt("How are models trained?", chunks)
    answer = generate_answer(prompt, temperature=0.1, max_tokens=200)
    
    print(f"Generated using convenience function")
    print(f"Answer length: {len(answer)} chars")
    print("Test 6 PASSED")
except Exception as e:
    print(f"Test 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All LLM client tests completed!")
print("="*60)