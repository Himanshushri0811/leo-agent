"""Verify DB connection, pgvector, existing tables, and alembic revision."""
import sys

# --- 1. psycopg2 direct ---
try:
    import psycopg2
    c = psycopg2.connect(
        host="127.0.0.1", port=5433, dbname="leo_db",
        user="leo_user", password="leo_password", connect_timeout=5
    )
    cur = c.cursor()
    cur.execute("SELECT version(), current_database()")
    ver, db = cur.fetchone()
    print(f"[OK] psycopg2 connected: {ver[:55]} | db={db}")

    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname='vector'")
    ext = cur.fetchone()
    print(f"[OK] pgvector: {ext}")

    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname='public'
        ORDER BY tablename
    """)
    tables = [r[0] for r in cur.fetchall()]
    print(f"[OK] Tables ({len(tables)}): {tables}")

    cur.execute("SELECT version_num FROM alembic_version")
    rev = cur.fetchone()
    print(f"[OK] Alembic revision: {rev[0] if rev else 'NONE'}")

    c.close()
except Exception as e:
    print(f"[FAIL] psycopg2: {type(e).__name__}: {e}")
    sys.exit(1)

# --- 2. asyncpg ---
import asyncio, asyncpg

async def test_asyncpg():
    try:
        conn = await asyncpg.connect(
            "postgresql://leo_user:leo_password@127.0.0.1:5433/leo_db",
            timeout=5
        )
        ver = await conn.fetchval("SELECT version()")
        print(f"[OK] asyncpg connected: {ver[:55]}")
        await conn.close()
    except Exception as e:
        print(f"[FAIL] asyncpg: {type(e).__name__}: {e}")

asyncio.run(test_asyncpg())
print("\nAll checks done.")
