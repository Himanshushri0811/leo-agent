# LEO AI Backend — Security Architecture

## 1. Threat Model & Mitigations

### 1.1 Prompt Injection Defense
- **Treat Retrieved Documents as Untrusted Data**: Documents and web pages ingested into the knowledge base are strictly parsed as data. System instructions explicitly prohibit the LLM from executing commands, tool invocations, or role transitions embedded in retrieved text.
- **Strict Boundary Delimitation**: Retrieved passages are segregated inside standard markers (`--- BEGIN KNOWLEDGE PASSAGES ---` and `--- END KNOWLEDGE PASSAGES ---`).
- **Prompt Protection**: The system prompt instructs LEO to decline attempts to extract instructions, system prompts, or configuration parameters.

### 1.2 Authentication & Authorization
- **Admin Endpoint Protection**: All ingestion, registration, and approval endpoints enforce `X-API-Key` validation via FastAPI security dependencies. Unauthorized requests return `401 Unauthorized` or `403 Forbidden`.
- **Caller Decoupling**: The chat endpoint is stateless from the perspective of user identities; caller platforms pass a `session_id`, isolating session context per caller.

### 1.3 Denial of Service & Resource Exhaustion
- **Payload Size Enforcement**: Middleware enforces a 1 MB limit on request bodies (`413 Request Entity Too Large`).
- **Bounded History Turns**: History context is bounded to 6 round-trip turns (12 messages) maximum, preventing token budget runaway.
- **Provider Retries & Timeouts**: LLM calls are bound to configurable timeouts (default 30 seconds) and bounded retry attempts (3 retries with exponential backoff) using `tenacity`.

### 1.4 Data Privacy & Leak Prevention
- **Structured Log Masking**: Raw user question content and raw assistant generation text are excluded from standard access logs. Logs only record message length, token counts, session IDs, and request correlation IDs.
- **Safe Error Responses**: Unhandled exceptions log the stack trace with a unique `error_id` and return a generic 500 JSON response. Internal tracebacks are never exposed to API consumers.

### 1.5 Database Security
- **Least Privilege Access**: Dedicated PostgreSQL user (`leo_user`) granted access only to the `leo_db` database.
- **Connection Pre-Ping**: Database connection pools are validated before acquisition to avoid executing on stale connections.
