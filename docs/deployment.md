# LEO AI Backend — Deployment & Operations Guide

## 1. Local Development with Docker Compose

Run the entire stack (FastAPI backend + PostgreSQL 16 with pgvector) locally:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Start services
docker compose up -d

# 3. Apply database migrations
docker compose exec app alembic upgrade head

# 4. (Optional) Run controlled knowledge seeding
docker compose exec app python scripts/seed_knowledge.py

# 5. Check logs
docker compose logs -f app
```

The API will be available at `http://localhost:8000`.
Swagger UI documentation (development only) at `http://localhost:8000/docs`.

---

## 2. Production Container Deployment

### 2.1 Multi-Stage Slim Dockerfile
The production Docker image runs as a non-root user (`appuser`), strips unnecessary compilers after installation, and starts Uvicorn with graceful signal handling:

```bash
docker build -t leo-ai-backend:v0.1.0 .
```

### 2.2 Environment Variables (Production)
Configure the following via your orchestrator (Kubernetes Secrets / AWS Secrets Manager / HashiCorp Vault):
- `APP_ENV`: `production`
- `APP_SECRET_KEY`: High-entropy 64-character secret
- `ADMIN_API_KEY`: Strong random API key for management endpoints
- `DATABASE_URL`: `postgresql+asyncpg://<user>:<password>@<db-host>:5432/leo_db`
- `LLM_PROVIDER`: `openai` or `anthropic`
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: Provider access credentials
- `EMBEDDINGS_PROVIDER`: `openai`

### 2.3 Container Orchestration Probes
- **Liveness Probe**: `GET /health/live` (initial delay 5s, period 10s)
- **Readiness Probe**: `GET /health/ready` (initial delay 10s, period 10s, checks DB connectivity)

### 2.4 Database Migrations in CI/CD
Run migrations as a discrete pipeline pre-step or Kubernetes `Job`:
```bash
alembic upgrade head
```
Do NOT auto-run destructive operations or migrations blindly inside container startup loops in high-availability clusters.
