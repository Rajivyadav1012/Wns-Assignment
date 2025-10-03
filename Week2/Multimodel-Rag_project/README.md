# Advanced RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with multiple retrieval strategies, memory management, and observability features.

![System Architecture](./screenshots/architecture-flow.png)
*Complete system flow diagram showing document processing, RAG initialization, and query handling*

## Overview

This RAG system provides an intelligent document querying interface with support for multiple retrieval methods, conversation memory, guardrails, and web search integration. Built with **Streamlit** for an intuitive user experience.

## Screenshots

### Main Interface
![Main Interface](./screenshots/main-interface.png)
*Built with **Streamlit** - Shows chat interface, document upload panel, and RAG configuration options*

### Document Upload & Processing
![Document Upload](./screenshots/document-upload.png)
*Document processing pipeline using **PyPDF2** (PDF), **python-docx** (DOCX), and **Tesseract OCR** (Images)*

### RAG Configuration
![RAG Configuration](./screenshots/rag-config.png)
*RAG type selection and feature toggles - **FAISS** for vector search, **rank-bm25** for keyword search*

### Query Results with Sources
![Query Results](./screenshots/query-results.png)
*Answer display with source attribution and similarity scores from **all-MiniLM-L6-v2** embeddings*

### Statistics Dashboard
![Statistics](./screenshots/statistics.png)
*System metrics showing document count, message history, and API status monitoring*

## Key Features

- **Multiple RAG Strategies**
  - Standard RAG (Vector-based semantic search)
  - Hybrid RAG (Combines vector + BM25 keyword search)
  - Knowledge Graph RAG (Entity-based retrieval)

- **Advanced Capabilities**
  - Conversation memory across sessions
  - Content guardrails for safe interactions
  - Web search integration for up-to-date information
  - Document chunking with configurable overlap
  - Multi-format document support (PDF, DOCX, TXT, Images)

- **Observability**
  - LangSmith integration for tracing and monitoring
  - Query metrics and performance tracking
  - Source attribution for answers

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Embeddings** | all-MiniLM-L6-v2 | Text-to-vector conversion (384 dimensions) |
| **Vector Database** | FAISS | Fast similarity search and clustering |
| **Metadata Storage** | SQLite | Document metadata and conversation history |
| **Text Extraction** | PyPDF2, python-docx | PDF and Word document parsing |
| **OCR Engine** | Tesseract | Image-to-text conversion |
| **Keyword Search** | BM25 (rank-bm25) | Traditional keyword-based retrieval |
| **Text Chunking** | LangChain TextSplitter | Document segmentation with overlap |
| **Web Search** | Serper API | Real-time web information retrieval |
| **LLM Providers** | Groq, OpenAI, Anthropic | Answer generation |
| **Guardrails** | Custom implementation | Content safety filtering |
| **Observability** | LangSmith | Tracing, logging, and monitoring |
| **Entity Extraction** | spaCy (en_core_web_sm) | Named entity recognition for KG RAG |
| **Graph Database** | NetworkX | Knowledge graph storage and traversal |

## System Architecture

### Document Processing Pipeline

![Document Processing](./screenshots/document-processing-detail.png)
*Detailed view of the text extraction and embedding process*

1. **Upload** - Support for PDF, DOCX, TXT, and image files
   - **Tool**: Streamlit file_uploader
   
2. **Text Extraction** 
   - **PDFs**: PyPDF2 library for text extraction
   - **DOCX**: python-docx for Word documents
   - **TXT**: Native Python file handling
   - **Images**: Tesseract OCR (pytesseract wrapper)
   
3. **Chunking** 
   - **Tool**: LangChain RecursiveCharacterTextSplitter
   - **Config**: 1000 characters per chunk with 200 character overlap
   - **Strategy**: Splits on paragraphs, sentences, then characters
   
4. **Embedding Generation**
   - **Model**: sentence-transformers/all-MiniLM-L6-v2
   - **Output**: 384-dimensional dense vectors
   - **Speed**: ~500 sentences/second on CPU
   
5. **Vector Storage**
   - **Database**: FAISS (Facebook AI Similarity Search)
   - **Index Type**: IndexFlatL2 for exact search
   - **Distance Metric**: L2 (Euclidean distance)
   
6. **Metadata Storage**
   - **Database**: SQLite
   - **Schema**: document_id, filename, chunk_index, text, timestamp
   - **Purpose**: Source attribution and retrieval

### Query Processing Flow

![Query Flow](./screenshots/query-processing.png)
*Step-by-step query handling from input to answer generation*

#### 1. Input Validation
- **Tool**: Custom guardrails implementation
- **Checks**: Toxicity, PII, prompt injection
- **Action**: Block or allow query processing

#### 2. Query Embedding
- **Model**: all-MiniLM-L6-v2 (same as documents)
- **Output**: 384-dimensional query vector

#### 3. Document Retrieval (Strategy-Specific)

**Standard RAG**
- **Tool**: FAISS similarity search
- **Method**: `index.search(query_vector, k=top_k)`
- **Threshold**: Similarity score > 0.3
- **Output**: Top-K most relevant chunks

**Hybrid RAG**
- **Semantic Search**: FAISS vector similarity
- **Keyword Search**: BM25 algorithm (rank-bm25 library)
- **Fusion**: Reciprocal Rank Fusion (RRF)
  - Formula: `score = Σ(1 / (k + rank_i))` where k=60
- **Output**: Combined and re-ranked results

**Knowledge Graph RAG**
- **Entity Extraction**: spaCy NER (PERSON, ORG, GPE, DATE)
- **Graph Storage**: NetworkX directed graph
- **Traversal**: BFS/DFS to find connected entities
- **Combination**: Graph results + semantic search
- **Output**: Context-enriched document chunks

#### 4. Web Search Enhancement (Optional)
- **API**: Serper (Google Search API wrapper)
- **Trigger**: When document results < threshold
- **Integration**: Appends web snippets to context
- **Limit**: Top 3-5 web results

#### 5. Context Building
- **Tool**: Custom prompt builder
- **Format**: System prompt + retrieved context + user query
- **Max tokens**: Configurable based on LLM limits

#### 6. LLM Generation
- **Providers**: 
  - **Groq**: Fast inference (LLaMA, Mixtral models)
  - **OpenAI**: GPT-3.5-turbo, GPT-4 models
  - **Anthropic**: Claude 3 models
- **Parameters**: Temperature, max_tokens, top_p configurable
- **Streaming**: Supported for real-time response display

#### 7. Output Validation
- **Tool**: Custom guardrails
- **Checks**: Toxicity, harmful content, hallucination detection
- **Action**: Filter or modify response

#### 8. Memory Storage
- **Database**: SQLite
- **Schema**: session_id, role (user/assistant), content, timestamp
- **Purpose**: Conversation context for follow-up queries

#### 9. Observability
- **Tool**: LangSmith
- **Logged Data**: Query, context, response, latency, token count
- **Visualization**: Trace viewer in LangSmith dashboard

### RAG Architecture Comparison

![RAG Comparison](./screenshots/rag-types-comparison.png)
*Visual comparison of Standard, Hybrid, and Knowledge Graph RAG approaches*

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm

# Install Tesseract OCR
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows - Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Requirements.txt

```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4  # Use faiss-gpu for GPU support
PyPDF2>=3.0.0
python-docx>=0.8.11
pytesseract>=0.3.10
rank-bm25>=0.2.2
langchain>=0.1.0
langsmith>=0.0.87
spacy>=3.7.0
networkx>=3.1
requests>=2.31.0
python-dotenv>=1.0.0
sqlite3  # Built-in with Python
```

## Configuration

Create a `.env` file or configure through the UI:

```env
# LLM Provider API Keys (at least one required)
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional Services
SERPER_API_KEY=your_serper_key  # For web search

# LangSmith Observability (optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=rag-system
LANGSMITH_TRACING=true

# System Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=5
```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Quick Start Guide

![Quick Start](./screenshots/quick-start-guide.png)
*Step-by-step visual guide for first-time users*

1. **Configure API Keys** 
   - Navigate to sidebar settings
   - Add your LLM provider API key (Groq, OpenAI, or Anthropic)
   - Optionally add Serper API key for web search

2. **Upload Documents** 
   - Click "Upload Documents" tab
   - Select PDF, DOCX, TXT files or images (PNG, JPG)
   - Wait for processing (progress bar shows status)
   - Uses: PyPDF2, python-docx, Tesseract OCR

3. **Initialize RAG** 
   - Choose RAG type: Standard, Hybrid, or Knowledge Graph
   - Toggle optional features: Memory, Guardrails, Web Search
   - Click "Initialize RAG"
   - Builds: FAISS index, BM25 index (if Hybrid), NetworkX graph (if KG)

4. **Ask Questions** 
   - Type your query in the chat input
   - View answer with source citations
   - Check metrics: retrieval time, token count, similarity scores

### RAG Type Selection

![RAG Selection](./screenshots/rag-selection-detail.png)
*Interactive guide for choosing the right RAG type*

**Standard RAG**
- **Best for**: General document querying, semantic understanding
- **Uses**: FAISS vector search with all-MiniLM-L6-v2 embeddings
- **Speed**: Fast (~50-100ms retrieval)
- **Accuracy**: High for semantic queries

**Hybrid RAG**
- **Best for**: Queries needing both semantic and exact keyword matching
- **Uses**: FAISS + BM25 with Reciprocal Rank Fusion
- **Speed**: Moderate (~100-200ms retrieval)
- **Accuracy**: Excellent for diverse query types

**Knowledge Graph RAG**
- **Best for**: Entity-relationship queries, connected information
- **Uses**: spaCy NER + NetworkX graph + FAISS
- **Speed**: Slower (~200-400ms retrieval)
- **Accuracy**: Superior for multi-hop reasoning

## Features in Detail

### Memory Management
![Memory Feature](./screenshots/memory-management.png)
- **Storage**: SQLite database with session management
- **Persistence**: Conversation history across sessions
- **Context Window**: Last N messages (configurable)
- **Tools**: Custom SQLite wrapper for CRUD operations

### Guardrails
![Guardrails](./screenshots/guardrails-feature.png)
- **Input Validation**: Toxicity detection, PII filtering
- **Output Filtering**: Harmful content detection
- **Implementation**: Rule-based + ML-based classifiers
- **Thresholds**: Configurable per category

### Web Search Integration
![Web Search](./screenshots/web-search-integration.png)
- **Trigger**: When document similarity < threshold
- **API**: Serper API (Google Search wrapper)
- **Processing**: Extract snippets, combine with document context
- **Display**: Separate source attribution for web results

### Observability with LangSmith
![LangSmith Dashboard](./screenshots/langsmith-tracing.png)
- **Tracing**: Full query execution path
- **Metrics**: Latency, token usage, cost estimation
- **Debugging**: Step-by-step inspection
- **Analytics**: Aggregate performance statistics

## Project Structure

```
rag-system/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── components/
│   ├── document_processor.py  # PyPDF2, python-docx, Tesseract
│   ├── embeddings.py          # all-MiniLM-L6-v2 wrapper
│   ├── vector_store.py        # FAISS operations
│   ├── rag_standard.py        # Standard RAG implementation
│   ├── rag_hybrid.py          # Hybrid RAG with BM25
│   ├── rag_knowledge_graph.py # KG RAG with spaCy + NetworkX
│   ├── memory_manager.py      # SQLite conversation storage
│   ├── guardrails.py          # Content safety filters
│   ├── web_search.py          # Serper API integration
│   └── llm_interface.py       # Multi-provider LLM wrapper
├── data/
│   ├── faiss_index/           # FAISS vector database
│   ├── metadata.db            # SQLite metadata storage
│   └── memory.db              # SQLite conversation history
├── screenshots/
│   ├── architecture-flow.png
│   ├── main-interface.png
│   ├── document-upload.png
│   ├── rag-config.png
│   ├── query-results.png
│   ├── statistics.png
│   ├── document-processing-detail.png
│   ├── query-processing.png
│   ├── rag-types-comparison.png
│   ├── quick-start-guide.png
│   ├── rag-selection-detail.png
│   ├── memory-management.png
│   ├── guardrails-feature.png
│   ├── web-search-integration.png
│   └── langsmith-tracing.png
└── README.md                  # This file
```

## Performance Consider