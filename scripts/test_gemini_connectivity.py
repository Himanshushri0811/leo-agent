"""
Isolated Gemini connectivity test and diagnostic utility for LEO.

Verifies API reachability, model accessibility, authentication, latency,
and error classification without leaking secrets, API keys, or auth headers.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from google import genai
from google.genai import errors, types
import httpx

from app.core.config import Settings, get_settings

_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}|[a-zA-Z0-9_-]{32,}")


def sanitize_message(msg: str) -> str:
    """Strip secret tokens or API keys from output strings."""
    return _KEY_PATTERN.sub("[REDACTED_API_KEY]", msg)


class DiagnosticCategory:
    SUCCESS = "SUCCESS"
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER_ERROR = "SERVER_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    CLIENT_ERROR = "CLIENT_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class GeminiDiagnosticResult:
    model: str
    success: bool
    elapsed_seconds: float
    response_text: str | None
    http_status: int | None
    exception_class: str | None
    sanitized_error: str | None
    failure_category: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def categorize_error(exc: BaseException) -> tuple[str, int | None, str]:
    """Categorize exception into a diagnostic category, status code, and clean error."""
    clean_err = sanitize_message(str(exc))
    status_code: int | None = None

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return DiagnosticCategory.CONNECTION_TIMEOUT, None, clean_err
    if isinstance(exc, httpx.TimeoutException):
        return DiagnosticCategory.CONNECTION_TIMEOUT, None, clean_err
    if isinstance(exc, httpx.NetworkError):
        return DiagnosticCategory.CONNECTION_TIMEOUT, None, clean_err

    if isinstance(exc, errors.APIError):
        status_code = exc.code
        if exc.code == 429:
            return DiagnosticCategory.RATE_LIMIT, 429, clean_err
        if exc.code in (408, 504) or "DEADLINE_EXCEEDED" in clean_err:
            return DiagnosticCategory.PROVIDER_TIMEOUT, status_code, clean_err
        if exc.code in (401, 403):
            return DiagnosticCategory.AUTHENTICATION_ERROR, status_code, clean_err
        if exc.code == 404:
            return DiagnosticCategory.MODEL_NOT_FOUND, 404, clean_err
        if exc.code and 500 <= exc.code < 600:
            return DiagnosticCategory.SERVER_ERROR, status_code, clean_err
        if exc.code and 400 <= exc.code < 500:
            return DiagnosticCategory.CLIENT_ERROR, status_code, clean_err

    return DiagnosticCategory.UNKNOWN_ERROR, status_code, clean_err


async def run_gemini_connectivity_test(
    model: str | None = None,
    prompt: str = "Reply with exactly: LEO_OK",
    timeout_seconds: float = 15.0,
    settings: Settings | None = None,
) -> GeminiDiagnosticResult:
    """
    Execute an isolated live Gemini API call and record detailed safe diagnostic results.
    Never prints or returns the raw API key.
    """
    cfg = settings or get_settings()

    if not cfg.gemini_api_key:
        return GeminiDiagnosticResult(
            model=model or "unknown",
            success=False,
            elapsed_seconds=0.0,
            response_text=None,
            http_status=None,
            exception_class="ValueError",
            sanitized_error="GEMINI_API_KEY is not configured in settings or environment.",
            failure_category=DiagnosticCategory.AUTHENTICATION_ERROR,
            details={"api_key_configured": False},
        )

    target_model = model or cfg.gemini_model or "gemini-3.5-flash"
    api_key = cfg.gemini_api_key.get_secret_value()

    # HttpOptions.timeout is in milliseconds
    timeout_ms = int(timeout_seconds * 1000)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )

    t0 = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model=target_model,
            contents=prompt,
        )
        elapsed = time.perf_counter() - t0
        text = response.text or ""

        return GeminiDiagnosticResult(
            model=target_model,
            success=True,
            elapsed_seconds=round(elapsed, 4),
            response_text=text.strip(),
            http_status=200,
            exception_class=None,
            sanitized_error=None,
            failure_category=DiagnosticCategory.SUCCESS,
            details={
                "api_key_configured": True,
                "timeout_ms": timeout_ms,
                "prompt_tokens": response.usage_metadata.prompt_token_count
                if response.usage_metadata
                else None,
                "candidate_tokens": response.usage_metadata.candidates_token_count
                if response.usage_metadata
                else None,
            },
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        category, status_code, clean_err = categorize_error(exc)
        return GeminiDiagnosticResult(
            model=target_model,
            success=False,
            elapsed_seconds=round(elapsed, 4),
            response_text=None,
            http_status=status_code,
            exception_class=type(exc).__name__,
            sanitized_error=clean_err,
            failure_category=category,
            details={
                "api_key_configured": True,
                "timeout_ms": timeout_ms,
            },
        )


async def main() -> int:
    settings = get_settings()
    print("=" * 70)
    print("LEO GEMINI CONNECTIVITY & DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"API Key Configured: {bool(settings.gemini_api_key)} (never logged or printed)")
    print(f"Configured Model in Settings: {settings.gemini_model}")
    print(f"LLM Timeout Setting: {settings.llm_timeout_seconds}s")
    print("-" * 70)

    # 1. Test gemini-2.5-flash to verify accessibility (as required by prompt)
    print("\n[Probe 1] Testing 'gemini-2.5-flash' accessibility...")
    res_25 = await run_gemini_connectivity_test(model="gemini-2.5-flash", settings=settings)
    print(f"  Category:       {res_25.failure_category}")
    print(f"  HTTP Status:    {res_25.http_status}")
    print(f"  Elapsed:        {res_25.elapsed_seconds}s")
    print(f"  Success:        {res_25.success}")
    if res_25.sanitized_error:
        print(f"  Sanitized Err:  {res_25.sanitized_error[:160]}...")

    # 2. Test configured / target model (e.g. gemini-3.5-flash)
    target = (
        settings.gemini_model if settings.gemini_model != "gemini-2.5-flash" else "gemini-3.5-flash"
    )
    print(f"\n[Probe 2] Testing target model '{target}'...")
    res_target = await run_gemini_connectivity_test(model=target, settings=settings)
    print(f"  Category:       {res_target.failure_category}")
    print(f"  HTTP Status:    {res_target.http_status}")
    print(f"  Elapsed:        {res_target.elapsed_seconds}s")
    print(f"  Success:        {res_target.success}")
    if res_target.response_text:
        print(f"  Response:       {res_target.response_text!r}")
    if res_target.sanitized_error:
        print(f"  Sanitized Err:  {res_target.sanitized_error[:160]}...")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  gemini-2.5-flash accessible: {res_25.success} ({res_25.failure_category})")
    print(f"  {target} accessible: {res_target.success} ({res_target.failure_category})")
    print("=" * 70)

    return 0 if res_target.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
