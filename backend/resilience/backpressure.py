import logging
from typing import Any

logger = logging.getLogger("neuroflow.resilience.backpressure")


async def get_ingestion_queue_depth(redis: Any) -> int:
    """Retrieve the current depth of the ingestion queue in Redis."""
    if not redis:
        return 0

    try:
        # Check standard queue:ingest or arq queue
        depth = await redis.llen("queue:ingest")
        if depth == 0:
            # Fallback to arq:queue:default if present
            arq_depth = await redis.zcard("arq:queue") or await redis.llen("arq:queue:default") or 0
            return max(int(depth or 0), int(arq_depth or 0))
        return int(depth or 0)
    except Exception as exc:
        logger.warning("Error reading ingestion queue depth: %s", exc)
        return 0


async def check_ingestion_backpressure(redis: Any) -> dict[str, Any]:
    """
    Evaluate ingestion backpressure based on queue depth.
    
    Rules:
    - queue_depth > 100 -> action: 'reject' (503), retry_after: 30
    - 50 < queue_depth <= 100 -> action: 'warn' (202), warning: 'high_queue_depth'
    - queue_depth <= 50 -> action: 'allow'
    """
    queue_depth = await get_ingestion_queue_depth(redis)

    if queue_depth > 100:
        return {
            "action": "reject",
            "status_code": 503,
            "payload": {
                "error": "ingestion_queue_full",
                "queue_depth": queue_depth,
                "retry_after": 30,
            },
        }

    if queue_depth > 50:
        estimated_wait_minutes = max(1, round(queue_depth * 0.1))
        return {
            "action": "warn",
            "status_code": 202,
            "warning": "high_queue_depth",
            "queue_depth": queue_depth,
            "estimated_wait_minutes": estimated_wait_minutes,
        }

    return {
        "action": "allow",
        "queue_depth": queue_depth,
    }
