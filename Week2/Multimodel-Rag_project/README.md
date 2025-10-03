# Advanced RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with multiple retrieval strategies, memory management, and observability features.

<img width="1827" height="980" alt="mermaid-diagram-2025-10-03-161329" src="https://github.com/user-attachments/assets/1b6ba98d-0e6a-4337-85f0-8f4716d4a3dc" />




---

## Overview

This RAG system provides an intelligent document querying interface with support for multiple retrieval methods, conversation memory, guardrails, and web search integration. Built with Streamlit for an intuitive user experience.

![WhatsApp Image 2025-10-03 at 10 06 47 AM](https://github.com/user-attachments/assets/abc3ad2a-05e8-4fc5-96af-aa5103fdfd7d)


---

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

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Embeddings** | all-MiniLM-L6-v2 | Text-to-vector (384 dimensions) |
| **Vector Database** | FAISS | Fast similarity search (IndexFlatL2) |
| **Database** | SQLite3 | Metadata & chat history (built-in Python) |
| **PDF Extraction** | PyPDF2 | PDF text extraction |
| **DOCX Extraction** | python-docx | Word document parsing |
| **Image Processing** | Pillow (PIL) | Image loading |
| **OCR Engine** | Tesseract | Image-to-text conversion |
| **Text Chunking** | LangChain | Document segmentation (RecursiveCharacterTextSplitter) |
| **Keyword Search** | rank-bm25 | BM25 algorithm (k1=1.5, b=0.75) |
| **NER** | spaCy (en_core_web_sm) | Named Entity Recognition |
| **Graph Database** | NetworkX | Knowledge graph storage |
| **Web Search** | Serper API | Google Search wrapper |
| **LLM Providers** | Groq, OpenAI, Anthropic | Answer generation |
| **Observability** | LangSmith | Tracing & monitoring |

---

## System Architecture

### Document Processing Pipeline

1. **Upload** - Streamlit file_uploader (PDF, DOCX, TXT, Images)
2. **Text Extraction** 
   - PDFs: PyPDF2.PdfReader
   - DOCX: python-docx.Document
   - Images: Tesseract OCR (pytesseract)
   - TXT: Native Python file reading
3. **Chunking** - LangChain RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. **Embedding** - all-MiniLM-L6-v2 generates 384-dimensional vectors
5. **Vector Storage** - FAISS IndexFlatL2 with L2 distance metric
6. **Metadata Storage** - SQLite3 stores document metadata and text
7. **Optional Indexing**:
   - Hybrid RAG: Build BM25 index (rank-bm25)
   - KG RAG: Extract entities (spaCy NER), build graph (NetworkX)


<img width="1080" height="579" alt="document" src="https://github.com/user-attachments/assets/a13105c7-ee38-434d-b886-43276386622f" />



### Query Processing Flow

1. **Input Validation** - Guardrails check for toxic/harmful content
2. **Query Embedding** - all-MiniLM-L6-v2 converts query to 384-dim vector
3. **Retrieval** - Strategy-specific document retrieval:
   - **Standard**: FAISS semantic search (similarity > 0.3, 50-100ms)
   - **Hybrid**: FAISS + BM25 with Reciprocal Rank Fusion (RRF k=60, 100-200ms)
   - **KG**: spaCy NER + NetworkX graph traversal + FAISS (60% graph + 40% semantic, 200-400ms)
4. **Web Enhancement** - Optional Serper API web search
5. **Context Building** - Format system prompt + retrieved docs + query
6. **LLM Generation** - Call Groq/OpenAI/Anthropic API with streaming support
7. **Output Validation** - Guardrails check response safety
8. **Memory Storage** - Save to SQLite3 conversation history
9. **Observability Logging** - Log trace to LangSmith with metrics
10. **Display** - Show answer with source citations and performance metrics

<img width="1080" height="579" alt="Query_Processing" src="https://github.com/user-attachments/assets/6064826b-f264-4b4a-8e55-3a806f4ea8a3" />


---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows - Download from https://github.com/UB-Mannheim/tesseract/wiki

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Requirements

```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
PyPDF2>=3.0.0
python-docx>=0.8.11
pytesseract>=0.3.10
Pillow>=10.0.0
rank-bm25>=0.2.2
langchain>=0.1.0
langsmith>=0.0.87
spacy>=3.7.0
networkx>=3.1
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## Configuration

Create a `.env` file in the project root:

```env
# ============================================
# LLM Provider API Keys (At least one required)
# ============================================
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# ============================================
# Optional Services
# ============================================
SERPER_API_KEY=your_serper_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=rag-system

# ============================================
# System Configuration
# ============================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=5

# BM25 Parameters (Hybrid RAG)
BM25_K1=1.5
BM25_B=0.75

# RRF Parameters (Hybrid RAG)
RRF_K=60

# Knowledge Graph Parameters
KG_MAX_DEPTH=2
KG_GRAPH_WEIGHT=0.6
KG_SEMANTIC_WEIGHT=0.4
```

**Alternative**: Configure via Streamlit UI sidebar settings

---

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`



---

### Quick Start Guide

#### Step 1: Configure API Keys

1. Open **Settings** in the sidebar
2. Enter at least one LLM provider API key:
   - **Groq**: Fast inference, affordable
   - **OpenAI**: High quality, GPT models
   - **Anthropic**: Long context, Claude models
3. Optionally add:
   - **Serper API**: For web search
   - **LangSmith API**: For observability
4. Click **Save Configuration**
---

#### Step 2: Upload Documents

1. Click **Upload Documents** tab in sidebar
2. Click **Browse files** button
3. Select files (PDF, DOCX, TXT, PNG, JPG)
4. Wait for processing (progress bar shows status)
5. View confirmation with document count

![WhatsApp Image 2025-10-03 at 10 07 17 AM](https://github.com/user-attachments/assets/0a861457-b54a-48c7-a60b-c2dfe8370d3f)



**Processing Details**:
- PDFs → PyPDF2 text extraction
- DOCX → python-docx paragraph extraction  
- Images → Tesseract OCR (may take longer)
- All chunks → Embedded with all-MiniLM-L6-v2 → Stored in FAISS + SQLite3

---

#### Step 3: Initialize RAG System

1. Select **RAG Type**:
   - **Standard RAG**: Fast semantic search (FAISS)
   - **Hybrid RAG**: Semantic + keyword (FAISS + BM25)
   - **Knowledge Graph RAG**: Entity relationships (spaCy + NetworkX)
2. Enable optional features:
   - ☑️ **Memory**: Conversation history (SQLite3)
   - ☑️ **Guardrails**: Content filtering
   - ☑️ **Web Search**: Serper API fallback
3. Click **Initialize RAG System**

![Rag_type](https://github.com/user-attachments/assets/0499760f-6972-4f5b-9fec-fe604cee0234)



---

#### Step 4: Ask Questions

1. Type your question in the chat input box
2. Press Enter or click Send
![before](https://github.com/user-attachments/assets/1527b35a-1658-4455-ae12-7ba7fdff8199)


3. View the response with:
   - **Answer**: LLM-generated response
   - **Sources**: Retrieved documents with similarity scores
   - **Metrics**: Retrieval time, token count, cost

![After](https://github.com/user-attachments/assets/b9a7c2c7-3d8d-4626-8535-8aa18189a95d)


4. Continue conversation (memory maintains context)

---

## RAG Type Selection

---

### Standard RAG
**How It Works**:
1. Query → 384-dim vector (all-MiniLM-L6-v2)
2. FAISS similarity search (L2 distance)
3. Filter results (similarity > 0.3)
4. Return top-K=5 documents

**Tools**: all-MiniLM-L6-v2 + FAISS IndexFlatL2

**Best For**:
- ✅ General question answering
- ✅ Semantic understanding
- ✅ Fast responses

---

### Hybrid RAG
**How It Works**:
1. Parallel search:
   - Path A: FAISS semantic search
   - Path B: BM25 keyword search (k1=1.5, b=0.75)
2. Reciprocal Rank Fusion: score = Σ(1 / (60 + rank))
3. Combine: 50% semantic + 50% keyword
4. Return top-K=5 merged results

**Tools**: FAISS + rank-bm25 + RRF

**Best For**:
- ✅ Mixed semantic + keyword queries
- ✅ Maximum accuracy
- ✅ Technical documents with specific terms

---

### Knowledge Graph RAG
**How It Works**:
1. Extract entities: spaCy NER (PERSON, ORG, GPE, DATE)
2. Graph traversal: NetworkX BFS (max depth=2)
3. Score nodes: direct=1.0, 1-hop=0.7, 2-hop=0.5
4. Fusion: 60% graph + 40% semantic (FAISS)
5. Return top-K=5 with entity metadata

**Tools**: spaCy en_core_web_sm + NetworkX + FAISS

**Best For**:
- ✅ Entity-relationship queries
- ✅ Multi-hop reasoning
- ✅ "Who knows who" questions

**Example Queries**:
- "Who worked with [person] at [company]?"
- "What companies are connected to [entity]?"

---

## Features in Detail
### Memory Management
- **Storage**: SQLite3 embedded database (Python built-in)
- **Persistence**: Conversation history across sessions
- **Context**: Last N messages used for follow-up queries
- **Implementation**: Automatic context injection

---

### Guardrails

- **Input Validation**: Toxicity detection, PII filtering, prompt injection
- **Output Filtering**: Harmful content detection
- **Implementation**: Rule-based + ML classifiers
- **Action**: Block and show warning message

---

### Web Search Integration

- **Trigger**: Automatically when document similarity < threshold
- **API**: Serper (Google Search wrapper)
- **Process**: Retrieve top 3-5 web results, merge with document context
- **Display**: Separate source attribution for web results

---

### Observability with LangSmith

**Tracked Metrics**:
- Query text and embeddings
- Retrieved documents with scores
- LLM provider and model
- Generated response
- Latency breakdown (retrieval, generation, total)
- Token counts and cost estimation

**Benefits**: Debug queries, optimize performance, track costs

---

## Project Structure

```
rag-system/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── README.md                       # This file
│
├── components/                     # Core modules
│   ├── document_processor.py      # PyPDF2, python-docx, Tesseract
│   ├── embeddings.py              # all-MiniLM-L6-v2 wrapper
│   ├── vector_store.py            # FAISS operations
│   ├── rag_standard.py            # Standard RAG
│   ├── rag_hybrid.py              # Hybrid RAG with BM25
│   ├── rag_knowledge_graph.py     # KG RAG with spaCy + NetworkX
│   ├── memory_manager.py          # SQLite3 conversation storage
│   ├── guardrails.py              # Content safety filters
│   ├── web_search.py              # Serper API integration
│   └── llm_interface.py           # Multi-provider LLM wrapper
│
├── data/                           # Data storage (auto-created)
│   ├── faiss_index/               # FAISS vector database
│   ├── metadata.db                # SQLite3 document metadata
│   └── memory.db                  # SQLite3 chat history
│
└── screenshots/                    # Documentation images
    └── (add your screenshots here)
```

---

## Performance Considerations

### Metrics

| Metric | Standard RAG | Hybrid RAG | Knowledge Graph RAG |
|--------|--------------|------------|---------------------|
| **Retrieval Time** | 50-100ms | 100-200ms | 200-400ms |
| **Total Response** | 1-2s | 1.5-2.5s | 2-3s |
| **Memory Usage** | Low | Moderate | High |
| **CPU Usage** | Low | Moderate | High |

**[SCREENSHOT: Performance Statistics]**

### Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Chunk Size** | 1000 characters | Balances context and precision |
| **Chunk Overlap** | 200 characters | Ensures continuity across boundaries |
| **Embedding Dimensions** | 384 | all-MiniLM-L6-v2 output size |
| **Similarity Threshold** | 0.3 | Filters irrelevant results (L2 distance) |
| **Top-K Retrieval** | 5 documents | Prevents context overload |
| **BM25 K1** | 1.5 | Term frequency saturation |
| **BM25 B** | 0.75 | Document length normalization |
| **RRF K** | 60 | Reciprocal rank fusion constant |

### Benchmarks

- **Embedding Speed**: ~500 sentences/sec (CPU), ~2000 sentences/sec (GPU)
- **FAISS Search**: <100ms for 100K vectors
- **Document Processing**: 2-5 seconds per document

---

## Troubleshooting

### Common Issues

**"API Key not configured"**
- Solution: Add LLM API key in `.env` file or UI settings

---

**"Tesseract not found"**
- Solution: Install Tesseract OCR for your OS
- Set `TESSERACT_CMD` environment variable if needed

---

**"FAISS index not initialized"**
- Solution: Upload documents first, then initialize RAG

---

**Poor retrieval results**
- Solution: Lower similarity threshold to 0.2
- Solution: Use Hybrid RAG for better accuracy
- Solution: Increase Top-K results to 10
- Solution: Enable web search

---

**Out of memory errors**
- Solution: Use FAISS IndexIVFFlat for large datasets
- Solution: Reduce chunk size to 500 characters
- Solution: Process documents in batches

---

**Slow performance**
- Solution: Use Standard RAG instead of Hybrid/KG
- Solution: Reduce Top-K to 3
- Solution: Use Groq for faster LLM inference

---

## API Keys

Get your API keys from:

| Provider | URL | Purpose |
|----------|-----|---------|
| **Groq** | https://console.groq.com/keys | Fast LLM inference |
| **OpenAI** | https://platform.openai.com/api-keys | GPT models |
| **Anthropic** | https://console.anthropic.com/settings/keys | Claude models |
| **Serper** | https://serper.dev/api-key | Web search |
| **LangSmith** | https://smith.langchain.com/settings | Observability |

---

## Limitations

- Requires API keys for LLM providers (Groq/OpenAI/Anthropic)
- OCR quality depends on image resolution (Tesseract limitation)
- Memory usage scales with document count (FAISS in-memory index)
- Web search requires Serper API subscription
- spaCy NER accuracy varies by domain
- Maximum context length limited by chosen LLM

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

---

## License

[Add your license here - MIT, Apache 2.0, etc.]

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review flowcharts for system behavior
- Check LangSmith traces for debugging

---

## Acknowledgments

- **Streamlit** - UI framework
- **sentence-transformers** - Embedding models (all-MiniLM-L6-v2)
- **FAISS** - Efficient similarity search (Meta AI Research)
- **LangChain** - Document processing utilities
- **spaCy** - NLP and named entity recognition
- **Tesseract** - OCR engine (Google)
- **NetworkX** - Graph algorithms
- **LangSmith** - Observability platform

---

## Citations

```bibtex
@article{reimers2019sentence,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  journal={arXiv preprint arXiv:1908.10084},
  year={2019}
}

@article{johnson2019billion,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  year={2019}
}
```

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Documentation**: Complete with flowcharts and guides

---

## Quick Reference

### Commands
```bash
# Start application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```



