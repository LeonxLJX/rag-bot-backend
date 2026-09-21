# RAG Bot Backend

**Enterprise-grade RAG (Retrieval-Augmented Generation) Platform**

Multi-tenant knowledge base system with hybrid search, reranking, conversation memory, and agent tools.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-yellow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Frontend)                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼───────────────────────────────────┐
│                        FastAPI Server                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Chat API │  │   KB API  │  │  Auth API │  │   Health Check   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                         Core Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ RAG Engine│  │  Reranker │  │   Chat   │  │   Agent Tools    │ │
│  │ (Hybrid)  │  │ (Cross-   │  │ (Memory) │  │  (KB Search,     │ │
│  │           │  │  Encoder) │  │          │  │   Calculator)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Infrastructure                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Chroma DB │  │  Redis    │  │PostgreSQL│  │   Background     │ │
│  │(Vector)   │  │  (Cache)  │  │  (Users)  │  │   Tasks (Celery)  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core RAG
- ✅ **Hybrid Search** — Dense vector + BM25 sparse retrieval
- ✅ **Reranking** — Cross-Encoder reranking (Top 20 → Top 5)
- ✅ **Guardrails** — Anti-hallucination checks
- ✅ **Caching** — Redis cache for frequent queries

### Multi-Tenant
- ✅ **Isolated Knowledge Bases** — Each user/organization has its own KB
- ✅ **User Management** — Registration, login, JWT auth
- ✅ **Role-based Access** — (coming soon)

### Conversation
- ✅ **Multi-turn Memory** — Chat history per session
- ✅ **Query Rewriting** — Follow-up questions → standalone questions
- ✅ **Streaming Output** — (coming soon)

### Agent
- ✅ **Tool Factory** — KB search, calculator, web search (stub)
- ✅ **Extensible** — Add custom tools easily

### Production Ready
- ✅ **Docker Support** — Dockerfile + docker-compose
- ✅ **Unit Tests** — Pytest
- ✅ **Structured Logging** — (coming soon)
- ✅ **Monitoring** — LangSmith integration (coming soon)

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/LeonxLJX/rag-bot-backend.git
cd rag-bot-backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the server
python main.py
```

Server runs at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api
```

---

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register` — Register new user
- `POST /api/auth/login` — Login, get JWT token

### Chat
- `POST /api/chat/` — Send a message
- `GET /api/chat/history/{session_id}` — Get chat history

### Knowledge Base
- `POST /api/knowledge-base/{kb_id}/upload` — Upload document
- `GET /api/knowledge-base/{kb_id}/stats` — Get KB stats
- `DELETE /api/knowledge-base/{kb_id}` — Delete KB

### Health
- `GET /health` — Health check

---

## 📁 Project Structure

```
rag-bot-backend/
├── main.py                    # FastAPI entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Docker Compose
├── .env.example               # Environment variables template
│
├── config/
│   └── settings.py            # Configuration (pydantic-settings)
│
├── api/
│   └── routes/
│       ├── chat.py            # Chat endpoints
│       ├── knowledge_base.py  # KB management
│       └── auth.py            # Auth endpoints
│
├── core/
│   ├── rag/
│   │   └── engine.py          # RAG engine
│   ├── ingest/
│   │   └── loader.py          # Document ingestion
│   ├── knowledge_base/
│   │   └── manager.py         # KB lifecycle management
│   ├── file_rag/
│   │   └── engine.py          # File RAG pipeline
│   ├── reranker/
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── chat/
│   │   └── chain.py           # Conversation chain with memory
│   ├── agent/
│   │   └── tools_factory.py    # Agent tools
│   ├── auth/
│   │   └── jwt_auth.py        # JWT authentication
│   ├── db/
│   │   └── models.py          # SQLAlchemy models
│   ├── cache/
│   │   └── redis_cache.py     # Redis caching
│   └── tasks/
│       └── document_tasks.py  # Background tasks
│
├── tests/
│   └── test_core.py           # Unit tests
│
└── data/
    ├── uploads/               # Uploaded files
    └── vector_db/              # Chroma vector stores
```

---

## ⚙️ Tuning Parameters

All tunable parameters in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `RETRIEVER_TOP_K` | 20 | Candidates to retrieve |
| `RERANK_TOP_N` | 5 | After reranking |
| `LLM_TEMPERATURE` | 0.2 | LLM creativity |
| `USE_HYBRID_SEARCH` | True | Vector + BM25 |
| `USE_RERANKER` | True | Cross-encoder rerank |
| `USE_GUARDRAIL` | True | Anti-hallucination |
| `USE_CACHE` | True | Redis cache |

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v
```

---

## 📜 License

MIT
