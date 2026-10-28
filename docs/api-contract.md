# LEO AI Backend — API Contract Specification

All request and response bodies use UTF-8 JSON. Every response includes the `X-Request-ID` header.

---

## 1. Chat Endpoint

### `POST /api/v1/chat`
Answers customer questions grounded in verified Bitmax Group company knowledge.

#### Headers
| Name | Type | Required | Description |
|---|---|---|---|
| `Content-Type` | `string` | Yes | Must be `application/json` |
| `X-Request-ID` | `string` | No | Optional client correlation identifier |

#### Request Body
```json
{
  "session_id": "sess_enterprise_9918",
  "message": "What cloud consulting services does Bitmax Group offer?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hi, I represent an enterprise looking for IT consulting."
    },
    {
      "role": "assistant",
      "content": "Hello! Bitmax Group provides a range of enterprise technology solutions. How can I assist you today?"
    }
  ]
}
```

#### Response (200 OK)
```json
{
  "session_id": "sess_enterprise_9918",
  "response": "Bitmax Group offers three core service pillars including Cloud Infrastructure & DevOps (multi-cloud migration, Kubernetes orchestration, and continuous delivery automation), Cybersecurity & Governance, and Custom Enterprise Engineering.",
  "sources": [
    {
      "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "source_name": "Bitmax Group - Core Services",
      "source_url": "https://bitmaxgroup.com/services",
      "similarity_score": 0.8841,
      "content_excerpt": "Bitmax Group offers three core service pillars: 1. Cloud Infrastructure & DevOps..."
    }
  ],
  "metadata": {
    "model": "gpt-4o-mini",
    "provider": "openai",
    "prompt_tokens": 420,
    "completion_tokens": 68,
    "prompt_version": "1.0.0",
    "sources_count": 1
  }
}
```

---

## 2. Knowledge Management Endpoints (Admin Only)

All knowledge management endpoints require the `X-API-Key` header matching `ADMIN_API_KEY`.

### `POST /api/v1/knowledge/sources`
Registers a source into `pending` state.

```json
// Request
{
  "name": "Bitmax Group - Security Whitepaper",
  "source_url": "https://bitmaxgroup.com/whitepapers/security.pdf",
  "notes": "Verified 2026 enterprise compliance overview"
}

// Response (201 Created)
{
  "source_id": "e6a2b8e3-0d65-4f76-b605-7f41cb838aa4",
  "name": "Bitmax Group - Security Whitepaper",
  "status": "pending",
  "message": "Source registered. Use /ingest to process it."
}
```

### `POST /api/v1/knowledge/ingest`
Fetches, parses, chunks, and embeds content, storing in `review` status.

```json
// Request
{
  "source_url": "https://bitmaxgroup.com/about",
  "name": "Bitmax Group - About Us"
}

// Response (200 OK)
{
  "source_id": "8f30c345-2be9-4a9f-a2e9-4e5029486c77",
  "source_name": "Bitmax Group - About Us",
  "chunks_created": 3,
  "is_duplicate": false,
  "status": "review",
  "message": "Ingested 3 chunks. Awaiting review and approval."
}
```

### `POST /api/v1/knowledge/sources/{id}/approve`
Promotes a source from `review` to `approved` status, enabling it for live RAG retrieval.

```json
// Request
{
  "approved_by": "lead-reviewer@bitmaxgroup.com"
}

// Response (200 OK)
{
  "id": "8f30c345-2be9-4a9f-a2e9-4e5029486c77",
  "name": "Bitmax Group - About Us",
  "source_type": "url",
  "source_url": "https://bitmaxgroup.com/about",
  "status": "approved",
  "version": 1,
  "created_at": "2026-10-19T10:00:00Z",
  "approved_at": "2026-10-19T10:05:00Z",
  "approved_by": "lead-reviewer@bitmaxgroup.com"
}
```

---

## 3. Health & Diagnostic Endpoints

### `GET /health/live`
- Returns `200 OK` when process is running: `{"status": "ok", "version": "0.1.0"}`

### `GET /health/ready`
- Returns `200 OK` when DB is reachable, or `503 Service Unavailable` if database check fails:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "checks": {
    "database": "ok"
  }
}
```

### `GET /metrics`
- Exposes Prometheus text format metrics for scrapers.
