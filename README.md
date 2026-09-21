# RAG Bot Backend

Enterprise-grade RAG (Retrieval-Augmented Generation) platform with multi-tenant knowledge bases.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (Frontend)                     │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────┐
│                    FastAPI Server                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Chat API   │  │    KB API   │  │  Health Check   │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    RAG Engine                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Retriever  │  │  Generator  │  │    Guardrail    │ │
│  │  (Hybrid)   │  │    (LLM)    │  │  (Anti-Halluc)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    Data Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Chroma DB  │  │    BM25     │  │  Document Store │ │
│  │  (Vector)   │  │  (Sparse)   │  │   (Files)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-tenant Knowledge Bases** — Isolated vector stores per tenant
- **Hybrid Search** — Dense vector + BM25 sparse retrieval
- **Document Ingestion** — PDF, TXT, Markdown support
- **Guardrails** — Anti-hallucination checks
- **REST API** — Clean, documented endpoints
- **CORS Enabled** — Ready for frontend integration

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key-here

# Run server
python main.py
```

Server runs at `http://localhost:8000`

---

## API Endpoints

### Chat
- `POST /api/chat/` — Send a message
- `GET /api/chat/history/{session_id}` — Get chat history

### Knowledge Base
- `POST /api/knowledge-base/{kb_id}/upload` — Upload document
- `GET /api/knowledge-base/{kb_id}/stats` — Get KB stats
- `DELETE /api/knowledge-base/{kb_id}` — Delete KB

### Health
- `GET /health` — Health check
- `GET /docs` — Swagger UI documentation

---

## Project Structure

```
rag-bot-backend/
├── main.py                  # FastAPI entry point
├── requirements.txt         # Dependencies
├── config/
│   └── settings.py          # Configuration
├── api/
│   ├── routes/
│   │   ├── chat.py          # Chat endpoints
│   │   └── knowledge_base.py # KB management
│   └── middleware/          # Auth, rate limiting
├── core/
│   ├── rag/
│   │   └── engine.py        # RAG engine
│   ├── ingest/
│   │   └── loader.py        # Document ingestion
│   └── models/              # Data models
└── data/
    ├── uploads/             # Uploaded files
    └── vector_db/           # Chroma vector stores
```

---

## Tuning Parameters

All tunable parameters are in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `RETRIEVER_TOP_K` | 20 | Number of candidates to retrieve |
| `RERANK_TOP_N` | 5 | Number after reranking |
| `LLM_TEMPERATURE` | 0.2 | LLM creativity (0 = factual) |
| `USE_HYBRID_SEARCH` | True | Enable vector + BM25 |
| `USE_GUARDRAIL` | True | Enable anti-hallucination |

---

## License

MIT
