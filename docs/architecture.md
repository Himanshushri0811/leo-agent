# LEO AI Backend — Architecture & Design

## 1. Architectural Philosophy

The LEO AI Backend is architected as a **Modular Monolith** in Python 3.12 with FastAPI. This pattern provides strict domain separation, fast iteration speed, straightforward local development, and simple containerized deployment without the premature complexity of microservices.

```
                           [ Calling Client / Platform ]
                                        ¦
                                        ? (HTTPS / JSON)
   +------------------------------------------------------------------------+
   ¦                            FastAPI Layer                               ¦
   ¦  - Request Correlation ID          - Structured Access Logging         ¦
   ¦  - Size Limit Validation           - Exception Handlers (Safe JSON)    ¦
   ¦  - Prometheus Instrumentation      - API Key Authentication            ¦
   +------------------------------------------------------------------------+
                    ¦                                       ¦
     +--------------?--------------+         +--------------?---------------+
     ¦         Chat Service        ¦         ¦      Knowledge Service       ¦
     +-----------------------------+         +------------------------------+
                    ¦                                       ¦
     +--------------?--------------+         +--------------?---------------+
     ¦       AI Orchestrator       ¦         ¦      Knowledge Pipeline      ¦
     ¦  - Context Trimming         ¦         ¦  - Parsing (HTML/PDF/Text)   ¦
     ¦  - Prompt Synthesis         ¦         ¦  - Chunking (Sliding Window) ¦
     ¦  - Provider Invocation      ¦         ¦  - Embeddings Generation     ¦
     +-----------------------------+         +------------------------------+
             ¦             ¦                                ¦
    +--------?------+ +----?-------------+          +-------?--------+
    ¦  LLM Adapter  ¦ ¦ Embeddings Adap. ¦          ¦ pgvector Store ¦
    ¦ (OpenAI/Anth/ ¦ ¦ (OpenAI / Mock)  ¦          ¦ (PostgreSQL)   ¦
    ¦  Mock)        ¦ +------------------+          +----------------+
    +---------------+
```

## 2. Core Subsystems

### 2.1 AI Provider Integration Layer (`app/ai/`)
- **Protocol-driven abstraction**: `LLMProvider` and `EmbeddingsProvider` protocols define strict input/output contracts.
- **Provider Adapters**:
  - `openai_provider.py`: Wraps OpenAI chat completions and embeddings with bounded exponential backoff (`tenacity`).
  - `anthropic_provider.py`: Adapts Claude Messages API, separating system instructions from conversation turns.
  - `mock.py`: Deterministic mock provider producing reproducible outputs without network dependencies for unit/CI tests.
- **System Prompts**: Maintained as version-controlled Python constants in `app/ai/prompts/system.py`.

### 2.2 Company Knowledge & RAG (`app/knowledge/`)
- **Parsing**: Strips HTML boilerplate and extracts PDF text into normalized UTF-8.
- **Deduplication**: Content SHA-256 hash checks prevent duplicate source entries.
- **Chunking**: Word-based sliding window with configurable overlap (`chunk_size=512`, `overlap=64`).
- **Semantic Retrieval**: Queries pgvector using cosine distance (`<=>`), strictly filtering by `ks.status = "approved"`.
- **Untrusted Content Isolation**: Retrieved passages are injected into an explicitly demarcated section in the prompt and accompanied by defensive instructions that treat passages strictly as untrusted data.

### 2.3 Data Persistence (`app/db/` & `app/repositories/`)
- **Async SQLAlchemy 2.x** with asyncpg for high-throughput asynchronous database operations.
- **pgvector** extension enables 1536-dimensional vector similarity operations directly inside PostgreSQL.
- **Repository Pattern**: Strict boundaries prevent leak of ORM sessions across business layers.

### 2.4 Observability (`app/observability/`)
- **Correlation IDs**: Generates or propagates `X-Request-ID` across every log entry via `structlog.contextvars`.
- **Metrics**: Exposes Prometheus metrics at `/metrics` (request counts, latency histograms, token usage, and retrieval latency).
