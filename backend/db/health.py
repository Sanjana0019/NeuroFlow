import logging
import time
from typing import Any

import asyncpg
import httpx
import redis.asyncio as redis

try:
    from backend.config import settings
    from backend.resilience.backpressure import get_ingestion_queue_depth
except ImportError:
    from config import settings
    from resilience.backpressure import get_ingestion_queue_depth

logger = logging.getLogger("neuroflow.health")


async def check_postgres(pool: asyncpg.Pool | None) -> tuple[bool, float | None]:
    """Check whether PostgreSQL is reachable and measure round-trip latency in ms."""
    if not pool:
        return False, None
    try:
        start = time.perf_counter()
        async with pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        latency_ms = (time.perf_counter() - start) * 1000.0
        return True, round(latency_ms, 2)
    except Exception as exc:
        logger.warning("Postgres health check failed: %s", exc)
        return False, None


async def check_redis(client: Any = None) -> tuple[bool, float | None]:
    """Check whether Redis is reachable and measure round-trip latency in ms."""
    try:
        if client is not None:
            start = time.perf_counter()
            await client.ping()
            latency_ms = (time.perf_counter() - start) * 1000.0
            return True, round(latency_ms, 2)
        else:
            r_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                decode_responses=True,
            )
            start = time.perf_counter()
            result = await r_client.ping()
            latency_ms = (time.perf_counter() - start) * 1000.0
            await r_client.aclose()
            return bool(result), round(latency_ms, 2)
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False, None


async def check_mlflow() -> tuple[bool, float | None]:
    """Check whether MLflow tracking server is reachable and measure latency in ms."""
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.mlflow_tracking_uri}/health")
            latency_ms = (time.perf_counter() - start) * 1000.0
            return response.is_success, round(latency_ms, 2)
    except Exception:
        return False, None


async def get_circuit_breakers_status(redis_client: Any = None) -> dict[str, dict[str, Any]]:
    """Retrieve state and failure count for all registered circuit breakers from Redis."""
    cb_status: dict[str, dict[str, Any]] = {
        "openai": {"state": "closed", "failure_count": 0}
    }

    if not redis_client:
        return cb_status

    try:
        # Check default openai circuit directly
        openai_state_raw = await redis_client.get("circuit:openai:state")
        if openai_state_raw:
            state = openai_state_raw.decode("utf-8") if isinstance(openai_state_raw, bytes) else str(openai_state_raw)
            fc_raw = await redis_client.get("circuit:openai:failure_count")
            failure_count = int(fc_raw) if fc_raw is not None else 0
            cb_status["openai"] = {"state": state, "failure_count": failure_count}

        # Scan for additional circuit state keys
        keys = []
        if hasattr(redis_client, "keys"):
            keys = await redis_client.keys("circuit:*:state")

        for k in keys:
            k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            parts = k_str.split(":")
            if len(parts) == 3:
                name = parts[1]
                state_raw = await redis_client.get(k_str)
                state = state_raw.decode("utf-8") if isinstance(state_raw, bytes) else str(state_raw or "closed")

                fc_raw = await redis_client.get(f"circuit:{name}:failure_count")
                failure_count = int(fc_raw) if fc_raw is not None else 0

                cb_status[name] = {
                    "state": state,
                    "failure_count": failure_count,
                }
    except Exception as exc:
        logger.warning("Error reading circuit breaker keys from Redis: %s", exc)

    return cb_status


async def perform_full_health_check(
    db_pool: asyncpg.Pool | None,
    redis_client: Any = None,
) -> dict[str, Any]:
    """
    Perform comprehensive health check returning overall status and detailed checks.
    
    Overall status rules:
    - critical: Postgres or Redis unreachable
    - degraded: Any circuit breaker is OPEN or MLflow unreachable
    - ok: All critical checks pass and no circuits are open
    """
    pg_ok, pg_latency = await check_postgres(db_pool)
    redis_ok, redis_latency = await check_redis(redis_client)
    mlflow_ok, mlflow_latency = await check_mlflow()
    circuits = await get_circuit_breakers_status(redis_client)
    queue_depth = await get_ingestion_queue_depth(redis_client)

    # Determine overall status
    if not pg_ok or not redis_ok:
        overall_status = "critical"
    elif any(c.get("state") == "open" for c in circuits.values()) or not mlflow_ok:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "checks": {
            "postgres": {
                "status": "ok" if pg_ok else "error",
                "latency_ms": pg_latency,
            },
            "redis": {
                "status": "ok" if redis_ok else "error",
                "latency_ms": redis_latency,
            },
            "mlflow": {
                "status": "ok" if mlflow_ok else "degraded",
                "latency_ms": mlflow_latency,
            },
            "circuit_breakers": circuits,
            "queue_depth": queue_depth,
            "worker_count": 2 if redis_ok else 0,
        },
    }