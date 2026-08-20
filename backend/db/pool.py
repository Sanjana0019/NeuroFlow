import asyncpg

from config import settings


async def create_pool() -> asyncpg.Pool:
    """Create the PostgreSQL connection pool."""
    return await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )