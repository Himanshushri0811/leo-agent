"""
Prometheus metrics definitions and instrumentation helpers.

Exposes standard metrics for:
- HTTP request count and latency
- LLM generation calls, latency, and token consumption
- Knowledge retrieval latency and result counts
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# HTTP metrics
HTTP_REQUESTS_TOTAL = Counter(
    "leo_http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "leo_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# LLM metrics
LLM_REQUESTS_TOTAL = Counter(
    "leo_llm_requests_total",
    "Total LLM generation calls",
    ["provider", "model", "status"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "leo_llm_request_duration_seconds",
    "LLM call latency in seconds",
    ["provider", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
)

LLM_TOKENS_TOTAL = Counter(
    "leo_llm_tokens_total",
    "Total tokens consumed by LLM calls",
    ["provider", "type"],  # type: "prompt" or "completion"
)

# Knowledge / RAG metrics
RETRIEVAL_DURATION_SECONDS = Histogram(
    "leo_retrieval_duration_seconds",
    "Knowledge retrieval latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

RETRIEVAL_RESULTS_COUNT = Histogram(
    "leo_retrieval_results_count",
    "Number of chunks retrieved per query",
    buckets=(0, 1, 2, 3, 5, 10),
)
