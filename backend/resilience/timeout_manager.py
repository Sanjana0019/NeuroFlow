import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger("neuroflow.resilience.timeout_manager")

T = TypeVar("T")

# Context-specific static timeouts (in seconds)
STATIC_TIMEOUTS: dict[str, float] = {
    "embedding": 60.0,
    "chat_completion": 60.0,
    "reranking": 15.0,
    "evaluation": 120.0,
    "file_extraction": 30.0,
    "url_fetch": 15.0,
}

# Bounds for adaptive timeouts (min_timeout, max_timeout)
TIMEOUT_BOUNDS: dict[str, tuple[float, float]] = {
    "embedding": (30.0, 180.0),
    "chat_completion": (30.0, 180.0),
    "reranking": (10.0, 60.0),
    "evaluation": (30.0, 300.0),
    "file_extraction": (15.0, 120.0),
    "url_fetch": (10.0, 60.0),
}

# Minimum observations required to enable adaptive timeout calculation
MIN_OBSERVATIONS_FOR_ADAPTIVE = 10
MAX_LATENCY_HISTORY = 1000


def calculate_percentile(values: list[float], percentile: float = 0.95) -> float:
    """Calculate percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]

    k = (n - 1) * percentile
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_v[f] + d * (sorted_v[c] - sorted_v[f])


class TimeoutManager:
    """Manages context-specific and adaptive timeouts with Redis latency tracking and trend analysis."""

    def __init__(self, redis: Any = None):
        self.redis = redis

    async def get_effective_timeout(self, task_type: str) -> float:
        """
        Calculate effective timeout for a task type.
        
        If sufficient observations exist in Redis (latency:{task_type}), computes p95 * 1.5 bounded
        by defined min/max bounds. Otherwise falls back to STATIC_TIMEOUTS.
        Also inspects 1-hour latency trends and emits a warning if latency is increasing.
        """
        base_timeout = STATIC_TIMEOUTS.get(task_type, 30.0)
        bounds = TIMEOUT_BOUNDS.get(task_type, (2.0, 180.0))

        if not self.redis:
            return base_timeout

        try:
            latency_key = f"latency:{task_type}"
            now = time.time()
            one_hour_ago = now - 3600.0

            # Retrieve all observations in sorted set (score is timestamp)
            entries = await self.redis.zrange(latency_key, 0, -1, withscores=True)
            if not entries or len(entries) < MIN_OBSERVATIONS_FOR_ADAPTIVE:
                return base_timeout

            all_latencies: list[float] = []
            recent_latencies: list[float] = []
            older_latencies: list[float] = []

            for member, score in entries:
                member_str = member.decode("utf-8") if isinstance(member, bytes) else str(member)
                # Member format: "timestamp:latency_seconds"
                if ":" in member_str:
                    try:
                        lat = float(member_str.split(":", 1)[1])
                        all_latencies.append(lat)
                        if score >= one_hour_ago:
                            recent_latencies.append(lat)
                        else:
                            older_latencies.append(lat)
                    except (ValueError, IndexError):
                        continue

            if len(all_latencies) < MIN_OBSERVATIONS_FOR_ADAPTIVE:
                return base_timeout

            # 1. Calculate overall p95
            p95 = calculate_percentile(all_latencies, 0.95)
            adaptive_timeout = p95 * 1.5

            # 2. Check 1-hour trend if sufficient recent and older observations exist
            if len(recent_latencies) >= 5 and len(older_latencies) >= 5:
                recent_p95 = calculate_percentile(recent_latencies, 0.95)
                older_p95 = calculate_percentile(older_latencies, 0.95)

                if older_p95 > 0 and (recent_p95 / older_p95) >= 1.25:
                    logger.warning(
                        "Increasing latency trend detected for task '%s' over the last hour: 1h p95=%.3fs vs baseline p95=%.3fs (+%.1f%%)",
                        task_type,
                        recent_p95,
                        older_p95,
                        ((recent_p95 / older_p95) - 1.0) * 100,
                    )

            # 3. Apply bounds
            min_bound, max_bound = bounds
            clamped_timeout = max(min_bound, min(max_bound, adaptive_timeout))
            return clamped_timeout

        except Exception as exc:
            logger.warning("Error computing adaptive timeout for '%s': %s (using static %ss)", task_type, exc, base_timeout)
            return base_timeout

    async def record_observation(self, task_type: str, elapsed_seconds: float) -> None:
        """Record completed task latency in Redis and trim to latest 1000 entries."""
        if not self.redis:
            return

        try:
            latency_key = f"latency:{task_type}"
            now = time.time()
            member = f"{now}:{elapsed_seconds}"

            # Add to sorted set with timestamp as score
            await self.redis.zadd(latency_key, {member: now})

            # Maintain only latest 1000 observations
            total = await self.redis.zcard(latency_key)
            if total > MAX_LATENCY_HISTORY:
                # Remove oldest elements
                await self.redis.zremrangebyrank(latency_key, 0, total - MAX_LATENCY_HISTORY - 1)
        except Exception as exc:
            logger.warning("Error recording latency observation for '%s': %s", task_type, exc)

    async def record_timeout(self, task_type: str) -> None:
        """Increment Redis timeout counter for the given task type."""
        if not self.redis:
            return

        try:
            await self.redis.incr(f"timeouts:{task_type}")
        except Exception as exc:
            logger.warning("Error recording timeout counter for '%s': %s", task_type, exc)

    async def execute(
        self,
        task_type: str,
        coro: Awaitable[T],
        override_timeout: float | None = None,
    ) -> T:
        """
        Execute an external coroutine with adaptive/static timeout and metrics recording.
        """
        timeout_seconds = override_timeout or await self.get_effective_timeout(task_type)
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            elapsed = time.perf_counter() - start_time
            await self.record_observation(task_type, elapsed)
            return result
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Task '%s' timed out after %.2f seconds (limit was %.2fs)",
                task_type,
                elapsed,
                timeout_seconds,
            )
            await self.record_timeout(task_type)
            raise TimeoutError(f"Task '{task_type}' timed out after {timeout_seconds:.2f} seconds") from None


_GLOBAL_TIMEOUT_MANAGER: TimeoutManager | None = None


def get_timeout_manager(redis: Any = None) -> TimeoutManager:
    """Get singleton TimeoutManager instance."""
    global _GLOBAL_TIMEOUT_MANAGER
    if _GLOBAL_TIMEOUT_MANAGER is None or (_GLOBAL_TIMEOUT_MANAGER.redis is None and redis is not None):
        _GLOBAL_TIMEOUT_MANAGER = TimeoutManager(redis=redis)
    return _GLOBAL_TIMEOUT_MANAGER


async def execute_with_timeout(
    task_type: str,
    coro: Awaitable[T],
    redis: Any = None,
    override_timeout: float | None = None,
) -> T:
    """Convenience helper to execute a coroutine with timeout management."""
    manager = get_timeout_manager(redis=redis)
    return await manager.execute(task_type, coro, override_timeout=override_timeout)
