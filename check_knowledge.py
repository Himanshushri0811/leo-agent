import asyncio
from sqlalchemy import text
from app.db.session import get_session_factory, init_db, dispose_db

async def main():
    init_db()
    try:
        factory = get_session_factory()

        queries = [
            ("Knowledge sources", """
                SELECT COUNT(*) AS total_sources
                FROM knowledge_sources
            """),
            ("Sources by status", """
                SELECT status, COUNT(*) AS source_count
                FROM knowledge_sources
                GROUP BY status
                ORDER BY status
            """),
            ("Knowledge chunks", """
                SELECT
                    COUNT(*) AS total_chunks,
                    COUNT(embedding) AS embedded_chunks,
                    COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings
                FROM knowledge_chunks
            """),
        ]

        async with factory() as session:
            for label, query in queries:
                print(f"\n--- {label} ---")
                result = await session.execute(text(query))
                for row in result.fetchall():
                    print(dict(row._mapping))
    finally:
        await dispose_db()

if __name__ == "__main__":
    asyncio.run(main())
