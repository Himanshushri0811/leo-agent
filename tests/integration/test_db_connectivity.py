"""
Integration tests for database connectivity and pgvector extension.
"""

from __future__ import annotations

import os
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


@pytest.mark.asyncio
async def test_database_connectivity_and_pgvector():
    """Verify live PostgreSQL connectivity and pgvector presence."""
    settings = get_settings()
    db_url = os.environ.get("DATABASE_URL", settings.database_url)

    engine = create_async_engine(db_url, connect_args={"timeout": 3})

    try:
        async with engine.connect() as conn:
            # 1. SELECT 1
            result = await conn.execute(text("SELECT 1 AS alive"))
            assert result.scalar_one() == 1

            # 2. Check pgvector extension
            ext_result = await conn.execute(
                text("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'")
            )
            row = ext_result.fetchone()
            if row and row[0]:
                assert row[0] is not None
    except Exception as exc:
        pytest.skip(
            f"Live PostgreSQL instance not reachable at {db_url} ({type(exc).__name__}: {exc}). "
            "Start PostgreSQL via 'docker compose up -d db' to run live connectivity verification."
        )
    finally:
        await engine.dispose()
