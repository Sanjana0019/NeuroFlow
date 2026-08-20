import asyncpg
import redis.asyncio as redis
import httpx

from config import settings


async def check_postgres(pool: asyncpg.Pool) -> bool:
    """Check whether PostgreSQL is reachable."""
    try:
        async with pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def check_redis() -> bool:
    """Check whether Redis is reachable."""
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
        )
        result = await client.ping()
        await client.aclose()
        return bool(result)
    except Exception:
        return False


async def check_mlflow() -> bool:
    """Check whether MLflow is reachable."""
    try:
        async with httpx.AsyncClient(
            timeout=3.0
        ) as client:
            response = await client.get(
                f"{settings.mlflow_tracking_uri}/health"
            )
            return response.is_success
    except Exception:
        return False