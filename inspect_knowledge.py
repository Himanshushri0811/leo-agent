import asyncio
from sqlalchemy import text
from app.db.session import init_db, get_session_factory, dispose_db

async def main():
    init_db()
    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(text("""
                SELECT
                    ks.name,
                    ks.source_type,
                    ks.source_url,
                    ks.status,
                    COUNT(kc.id) AS chunk_count,
                    COUNT(kc.embedding) AS embedded_chunks
                FROM knowledge_sources ks
                LEFT JOIN knowledge_chunks kc ON kc.source_id = ks.id
                GROUP BY ks.id, ks.name, ks.source_type,
                         ks.source_url, ks.status
                ORDER BY ks.name
            """))

            for row in result.mappings():
                print(dict(row))
    finally:
        await dispose_db()

if __name__ == "__main__":
    asyncio.run(main())
