### Conversational RAG Platform for YouTube

RAG system for conversational interaction with YouTube content

--- 

### Work in Progress...
This project is currently under active development. Repository structure has been set up and code implementation is in progress.

## Tech Stack

- **Framework**: LangChain
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Pinecone
- **Backend**: FastAPI
- **Frontend**: Chrome Extension (HTML/CSS/JavaScript)
- **Deployment**: Docker, Docker Compose

## Project Structure
```
conversational-rag-youtube/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── models.py
│   └── routes/
├── chains/                 # LangChain LCEL chains
│   ├── indexing_chain.py
│   └── qa_chain.py
├── indexing/              # Document loading and processing
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   └── vector_store.py
├── retrieval/             # Retrieval strategies
│   ├── simple_retriever.py
│   ├── query_rewriter.py
│   └── hybrid_retriever.py
├── augmentation/          # Prompt engineering
│   └── prompt_templates.py
├── generation/            # LLM generation
│   ├── llm_client.py
│   └── citation_handler.py
├── config/                # Configuration management
│   ├── settings.py
│   └── logging_config.py
├── utils/                 # Utilities
│   ├── exceptions.py
│   └── validators.py
├── chrome-extension/      # Chrome extension UI
├── tests/                 # Test suite
└── docs/                  # Documentation
```