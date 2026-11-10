import asyncio
import asyncpg

async def main():
    try:
        conn = await asyncpg.connect(
            'postgresql://leo_user:leo_password@127.0.0.1:5432/leo_db',
            timeout=5
        )
        ver = await conn.fetchval('SELECT version()')
        print(f'asyncpg OK: {ver[:60]}')
        await conn.close()
    except Exception as e:
        print(f'asyncpg FAILED: {type(e).__name__}: {e}')

asyncio.run(main())

