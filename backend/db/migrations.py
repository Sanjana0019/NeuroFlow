import asyncpg


async def check_schema(pool: asyncpg.Pool) -> bool:
    """Check whether the NeuroFlow database schema is present."""
    async with pool.acquire() as connection:
        return await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'documents'
            )
            """
        )