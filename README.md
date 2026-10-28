# LEO AI Backend

Production-grade, modular AI assistant backend for **Bitmax Group**, engineered for grounded customer conversational Q&A, approved knowledge retrieval (RAG), and calling-platform integration.

---

## 1. Architectural Highlights

- **Modular Monolith**: Clean layer separation across API, business services, AI orchestration, knowledge pipeline, and PostgreSQL database access.
- **Provider-Agnostic AI**: Unified protocol abstractions for LLMs (`OpenAI`, `Anthropic`, `Mock`) and Embeddings (`OpenAI`, `Mock`). Replaceable without touching business logic.
- **Controlled Knowledge Gate (RAG)**: Ingested documents enter `review` state and require administrative approval (`approved`) before becoming searchable by pgvector cosine similarity retrieval.
- **Prompt Injection Defense**: Retrieved passages are strictly treated as untrusted data between explicit delimiters, with system prompts enforcing boundary compliance.
- **Observability**: `structlog` structured logging, `X-Request-ID` correlation middleware, health probes (`/health/live`, `/health/ready`), and Prometheus metrics (`/metrics`).
- **DevSecOps Ready**: Non-root container (`appuser`), slim Python 3.12 image, bounded retries, size validation, and environment-driven secrets.

---

## 2. Directory Structure

```
leo-ai-backend/
+-- app/
¦   +-- main.py                     # FastAPI application factory & lifespan
¦   +-- api/
¦   ¦   +-- router.py               # Root API router
¦   ¦   +-- v1/
¦   ¦       +-- health.py           # Liveness & readiness probes
¦   ¦       +-- chat.py             # Customer chat endpoint
¦   ¦       +-- knowledge.py        # Knowledge sources, ingestion, & approval
¦   +-- core/
¦   ¦   +-- config.py               # Pydantic v2 settings & environment validation
¦   ¦   +-- logging.py              # Structured logging (JSON/Console) via structlog
¦   ¦   +-- exceptions.py           # Domain exceptions & safe JSON error handlers
¦   ¦   +-- security.py             # Admin API key auth & payload size checks
¦   +-- schemas/                    # Pydantic v2 request/response models
¦   +-- ai/
¦   ¦   +-- orchestrator.py         # Chat execution pipeline
¦   ¦   +-- interfaces.py           # LLM and Embeddings protocols
¦   ¦   +-- prompts/system.py       # Versioned system prompts with safety rules
¦   ¦   +-- providers/              # OpenAI, Anthropic, and Mock adapters
¦   +-- knowledge/
¦   ¦   +-- parsing.py              # HTML/PDF/Text extraction & normalisation
¦   ¦   +-- chunking.py             # Sliding-window text chunker
¦   ¦   +-- embeddings.py           # Batch embedding generation
¦   ¦   +-- retrieval.py            # pgvector cosine similarity query
¦   ¦   +-- ingestion.py            # Ingestion workflow
¦   +-- db/
¦   ¦   +-- session.py              # Async SQLAlchemy session factory & connection pool
¦   ¦   +-- base.py                 # Declarative ORM base
¦   ¦   +-- models/                 # Knowledge and Conversation entities
¦   +-- repositories/               # Encapsulated DB query logic
¦   +-- services/                   # Business domain services
¦   +-- observability/              # Prometheus metrics & correlation middleware
+-- knowledge_data/                 # Raw and processed offline knowledge store
+-- migrations/                     # Alembic migration scripts
+-- scripts/                        # Seeding and developer automation scripts
+-- tests/                          # Pytest suite (unit, api, security, evaluations)
+-- docs/                           # Architecture, API contract, security, deployment
+-- .env.example                    # Environment variable template
+-- compose.yaml                    # Local multi-container Docker Compose
+-- Dockerfile                      # Multi-stage production container build
+-- pyproject.toml                  # Pinned dependencies, Ruff, Mypy, & Pytest config
+-- Makefile                        # Unix developer commands
+-- tasks.ps1                       # Windows PowerShell developer tasks
+-- README.md
```

---

## 3. Implemented vs. Planned Work

| Component | Status | Details |
|---|---|---|
| **FastAPI Core & Lifespan** | **Implemented** | Async startup/shutdown, CORS, middleware, health endpoints |
| **API Endpoints** | **Implemented** | `/api/v1/chat`, `/api/v1/knowledge/sources`, `/api/v1/knowledge/ingest`, `/health/live`, `/health/ready`, `/metrics` |
| **AI Layer & Adapters** | **Implemented** | Abstract protocols, OpenAI adapter, Anthropic adapter, Mock adapter |
| **RAG & Knowledge Gate** | **Implemented** | Normalizer, chunker, batch embeddings, pgvector query, approval gate |
| **Security & Auth** | **Implemented** | Admin `X-API-Key` auth, prompt injection defense, 1MB size limit, safe 500 handlers |
| **Observability** | **Implemented** | Prometheus metrics, `X-Request-ID` correlation, structured logging |
| **Migrations & DB** | **Implemented** | SQLAlchemy 2.x async models, Alembic configuration, initial pgvector schema |
| **Docker & Compose** | **Implemented** | Multi-stage slim Dockerfile (non-root), Docker Compose with `pgvector:pg16` |
| **Test Suite** | **Implemented** | Unit, API, Security, and Evaluation grounding tests (all mock-compatible) |
| *SuperPhone / Twilio / Telephony* | *Out of Scope* | Excluded by design; calling platforms integrate via documented REST API |
| *Frontend / Admin Web UI* | *Out of Scope* | Excluded; headless API designed for external service integration |
| *Automated Web Crawler* | *Planned Future* | Controlled manual ingestion implemented; recursive crawler deferred |

---

## 4. Quick Start

### Option A: Local Development with Docker Compose (Recommended)

```bash
# 1. Clone/navigate to project
cd leo-ai-backend

# 2. Configure environment
cp .env.example .env

# 3. Start PostgreSQL with pgvector and API backend
docker compose up -d

# 4. Apply migrations
docker compose exec app alembic upgrade head

# 5. Seed initial verified company knowledge
docker compose exec app python scripts/seed_knowledge.py
```

The API is ready at `http://localhost:8000`.

### Option B: Local Python Development

**Prerequisites**: Python 3.12+, PostgreSQL with pgvector extension.

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run migrations
alembic upgrade head

# 4. Run tests
pytest

# 5. Launch development server
uvicorn app.main:app --reload --port 8000
```

---

## 5. Developer Task Reference

| Action | Linux / macOS | Windows PowerShell |
|---|---|---|
| Run tests | `make test` | `.\tasks.ps1 test` |
| Run linter | `make lint` | `.\tasks.ps1 lint` |
| Code formatting | `make format` | `.\tasks.ps1 format` |
| Static type check | `make typecheck` | `.\tasks.ps1 typecheck` |
| Run migrations | `make migrate` | `.\tasks.ps1 migrate` |
| Seed knowledge | `make seed` | `.\tasks.ps1 seed` |

---

## 6. Calling Platform Integration Contract

External platforms (e.g. telephony bridges, web chat widgets, mobile apps) integrate strictly via the versioned HTTP API:

- **Send a question**: `POST /api/v1/chat` with `{ "session_id": "...", "message": "..." }`
- **Receive response**: JSON containing `response`, `sources` (with chunk attribution and relevance score), and token consumption `metadata`.
- Detailed schema and payload examples: See [`docs/api-contract.md`](docs/api-contract.md).
