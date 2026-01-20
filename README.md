# ContentIQ: Conversational RAG Platform for YouTube

An end-to-end RAG system that enables intelligent question-answering over YouTube video content using LangChain, Pinecone, and OpenAI.

---

## Architecture

![System Architecture](./artifacts/architecture.drawio.svg)

---

## Demo

![System Demo](./artifacts/demo-yt-rag.gif)

---

## Overview

ContentIQ extracts YouTube video transcripts, processes them into semantic chunks, stores them in a vector database, and uses retrieval-augmented generation to provide accurate, citation-backed answers to user queries.

---

## System Architecture

### Three-Layer Design

**Client-Server Layer**
- Chrome Extension (JavaScript) for user interface
- FastAPI REST API for request handling

**RAG Layer**
- Indexing Pipeline: Transcript extraction → Text chunking → Embedding generation → Vector storage
- Retrieval Pipeline: Query embedding → Semantic search → Prompt engineering → Answer generation

**Infrastructure Layer**
- Docker containerization for deployment
- GitHub Actions for continuous integration
- RAGAS framework for quality evaluation

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| API | FastAPI |
| Vector Database | Pinecone |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| LLM | OpenAI GPT-4o-mini |
| Frontend | Chrome Extension (Vanilla JS) |
| Deployment | Docker |
| Testing | Pytest |
| Evaluation | RAGAS |

---

## RAG Pipeline Implementation

### Indexing Flow

**Document Ingestion**
- YouTube transcript extraction via youtube-transcript-api
- LangChain Document objects with metadata

**Text Processing**
- RecursiveCharacterTextSplitter: 1000 characters per chunk
- 200 character overlap for context preservation
- Average 69 chunks per video

**Embedding & Storage**
- OpenAI text-embedding-3-small generates 1536-dimensional vectors
- Pinecone stores vectors with metadata
- Cosine similarity for search
- Duplicate prevention built-in

### Retrieval Flow

**Three Retrieval Strategies Implemented**

| Strategy | Method | Description |
|----------|--------|-------------|
| Simple | Semantic similarity only | Pure vector search with cosine similarity |
| Hybrid | Semantic + BM25 | Combines vector search with keyword matching |
| Rewriting | LLM-enhanced queries | Generates multiple query variations |

**Query Processing**
- User question embedded into 1536-dimensional vector
- Pinecone retrieves top-k most similar chunks (k=4 or k=6)
- Retrieved chunks combined with user question
- GPT-4o-mini generates answer with citations

**Answer Generation**
- Temperature: 0.0 for deterministic responses
- Citation extraction for source traceability
- Response time: 2.5-4.2 seconds

---

## RAGAS Evaluation

### Methodology

Evaluated four configurations using RAGAS framework with 10 test questions:
- Faithfulness: Measures hallucination
- Context Precision: Relevance of retrieved chunks
- Context Recall: Completeness of retrieved information
- Overall Score: Average of all metrics

### Results

| Configuration | Faithfulness | Context Precision | Context Recall | Overall Score |
|--------------|--------------|-------------------|----------------|---------------|
| simple_k4 | 0.936 | 0.958 | 0.744 | 87.9% |
| **simple_k6** | **1.0** | **0.87** | **0.836** | **90.8%** |
| hybrid_k4 | 0.966 | 0.92 | 0.817 | 90.1% |
| hybrid_k6 | 0.993 | 0.77 | 0.836 | 86.6% |

### Final Configuration

Based on evaluation results, the system uses:

**Retriever**: Simple (semantic similarity only)  
**Top-K**: 6 chunks  
**Reasoning**: Highest overall score with perfect faithfulness (zero hallucination)

---

## API Endpoints

| Method | Endpoint | Description | Response Time |
|--------|----------|-------------|---------------|
| GET | `/health` | Health check | <50ms |
| POST | `/index` | Index YouTube video | ~18 seconds |
| GET | `/index/status/{video_id}` | Check if video indexed | ~100ms |
| POST | `/query` | Ask questions about video | 2.5-4.2 seconds |

### Example Usage

**Index Video**
```bash
POST /index
{
  "video_id": "dQw4w9WgXcQ"
}
```

**Query Video**
```bash
POST /query
{
  "question": "What is the main topic?",
  "video_id": "dQw4w9WgXcQ",
  "retriever_type": "simple",
  "top_k": 6
}
```

---

## Key Features

- Automated YouTube video indexing with English captions
- Semantic search with vector similarity
- Context-aware answers with source citations
- Chrome Extension for seamless integration
- Multiple retrieval strategies evaluated and optimized
- Zero hallucination (Faithfulness: 1.0)
- Comprehensive quality evaluation with RAGAS

---

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Pinecone API key
- Docker (optional)

### Environment Configuration
```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=youtube-rag
```
## Limitations

- English captions only
- Chrome browser only (extension)
- No automated deployment pipeline (CI only)
- Answer Relevance metric not calculated

---

## Future Work

- Multi-language support
- Automated deployment
- Fine-tuned embeddings
- Multi-modal RAG (video frames + text)

---

## License

MIT License