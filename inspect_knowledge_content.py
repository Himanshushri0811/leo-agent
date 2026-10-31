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
                    kc.chunk_index,
                    kc.content,
                    kc.chunk_metadata
                FROM knowledge_sources ks
                JOIN knowledge_chunks kc ON kc.source_id = ks.id
                ORDER BY ks.name, kc.chunk_index
            """))

            for row in result.mappings():
                print(f"\n{'=' * 60}")
                print("SOURCE:", row["name"])
                print("CHUNK:", row["chunk_index"])
                print("CONTENT:")
                print(row["content"])
                print("METADATA:", row["chunk_metadata"])
    finally:
        await dispose_db()

if __name__ == "__main__":
    asyncio.run(main())
