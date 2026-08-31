import asyncio
import asyncpg
from backend.config import settings

async def test():
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db
    )
    v_str = "[" + ",".join(["0.01"] * 2048) + "]"

    try:
        row = await conn.fetch("SELECT id, 1.0 - (c.embedding <=> $1::halfvec) as score FROM chunks c LIMIT 1", v_str)
        print("Query with $1::halfvec: SUCCESS!", row)
    except Exception as e:
        print("Error $1::halfvec:", e)

    try:
        row = await conn.fetch("SELECT id, 1.0 - (c.embedding <=> $1::vector) as score FROM chunks c LIMIT 1", v_str)
        print("Query with $1::vector: SUCCESS!", row)
    except Exception as e:
        print("Error $1::vector:", e)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(test())
