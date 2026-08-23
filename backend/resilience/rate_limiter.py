import asyncio
import logging
import time
from typing import Any
from uuid import UUID

logger = logging.getLogger("neuroflow.resilience.rate_limiter")


class TokenBucketRateLimiter:
    """Redis-backed Token Bucket Rate Limiter with atomic token consumption and continuous refill."""

    def __init__(
        self,
        key: str,
        capacity: float = 3000.0,
        refill_rate: float = 50.0,
        redis: Any = None,
    ):
        """
        :param key: Redis key prefix, e.g. 'rpb:openai' or 'rpb:pipeline:<id>'
        :param capacity: Maximum number of tokens the bucket can hold
        :param refill_rate: Tokens added per second
        :param redis: Redis client instance
        """
        self.key = key
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.redis = redis

        self.tokens_key = f"{key}:tokens"
        self.last_updated_key = f"{key}:last_updated"

        # In-memory fallback
        self._in_memory_tokens = self.capacity
        self._in_memory_last_updated = time.time()
        self._in_memory_lock = asyncio.Lock()

    async def _consume_in_memory(self, tokens: float) -> tuple[bool, float]:
        async with self._in_memory_lock:
            now = time.time()
            elapsed = now - self._in_memory_last_updated
            self._in_memory_last_updated = now
            self._in_memory_tokens = min(self.capacity, self._in_memory_tokens + elapsed * self.refill_rate)

            if self._in_memory_tokens >= tokens:
                self._in_memory_tokens -= tokens
                return True, 0.0
            else:
                needed = tokens - self._in_memory_tokens
                wait_time = needed / self.refill_rate if self.refill_rate > 0 else 1.0
                return False, wait_time

    async def _consume_redis(self, tokens: float) -> tuple[bool, float]:
        now = time.time()
        # Lua script to atomically refill and consume tokens
        lua_script = """
        local tokens_key = KEYS[1]
        local last_updated_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local current_tokens = tonumber(redis.call('get', tokens_key))
        local last_updated = tonumber(redis.call('get', last_updated_key))

        if current_tokens == nil then
            current_tokens = capacity
        end
        if last_updated == nil then
            last_updated = now
        end

        local elapsed = math.max(0, now - last_updated)
        current_tokens = math.min(capacity, current_tokens + (elapsed * refill_rate))

        if current_tokens >= requested then
            current_tokens = current_tokens - requested
            redis.call('set', tokens_key, tostring(current_tokens))
            redis.call('set', last_updated_key, tostring(now))
            return {1, 0}
        else
            local needed = requested - current_tokens
            local wait_time = needed / refill_rate
            redis.call('set', tokens_key, tostring(current_tokens))
            redis.call('set', last_updated_key, tostring(now))
            return {0, tostring(wait_time)}
        end
        """
        try:
            res = await self.redis.eval(
                lua_script,
                2,
                self.tokens_key,
                self.last_updated_key,
                str(self.capacity),
                str(self.refill_rate),
                str(tokens),
                str(now),
            )
            success = bool(res[0] == 1)
            wait_time = float(res[1]) if len(res) > 1 else 0.0
            return success, wait_time
        except Exception as exc:
            logger.warning("Redis Lua token bucket evaluation failed for '%s': %s (fallback in-memory)", self.key, exc)
            return await self._consume_in_memory(tokens)

    async def acquire(self, tokens: float = 1.0, timeout: float | None = 30.0) -> bool:
        """Acquire tokens from the bucket, waiting asynchronously without busy-looping if empty."""
        start_time = time.time()

        while True:
            if self.redis:
                success, wait_time = await self._consume_redis(tokens)
            else:
                success, wait_time = await self._consume_in_memory(tokens)

            if success:
                return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False

            sleep_duration = max(0.01, min(wait_time, 0.5))
            await asyncio.sleep(sleep_duration)

    async def __aenter__(self):
        acquired = await self.acquire(tokens=1.0)
        if not acquired:
            raise TimeoutError(f"Rate limiter '{self.key}' timed out waiting for token")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def get_global_llm_limiter(provider: str = "openai", redis: Any = None) -> TokenBucketRateLimiter:
    """Return the global LLM rate limiter (capacity=3000, refill=50/s)."""
    return TokenBucketRateLimiter(
        key=f"rpb:{provider}",
        capacity=3000.0,
        refill_rate=50.0,
        redis=redis,
    )


def get_pipeline_rate_limiter(
    pipeline_id: UUID | str,
    rate_limit_rpm: int = 60,
    redis: Any = None,
) -> TokenBucketRateLimiter:
    """Return a per-pipeline token bucket rate limiter based on rate_limit_rpm."""
    rpm = max(1, rate_limit_rpm)
    refill_rate = rpm / 60.0
    return TokenBucketRateLimiter(
        key=f"rpb:pipeline:{pipeline_id}",
        capacity=float(rpm),
        refill_rate=refill_rate,
        redis=redis,
    )


class EndpointRateLimiter:
    """Redis-backed sliding-window rate limiter per client IP."""

    @staticmethod
    async def check_rate_limit(
        redis: Any,
        client_ip: str,
        endpoint: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check whether client_ip has exceeded max_requests in the past window_seconds.
        
        Returns: (allowed: bool, retry_after: int)
        """
        if not redis:
            return True, 0

        key = f"ratelimit:{endpoint}:{client_ip}"
        now = time.time()
        window_start = now - window_seconds
        req_id = f"{now}:{time.perf_counter()}"

        try:
            # 1. Remove old entries outside the window
            await redis.zremrangebyscore(key, 0, window_start)

            # 2. Count requests in current window
            current_count = await redis.zcard(key)

            if current_count >= max_requests:
                # Find oldest timestamp to calculate retry_after
                oldest_entries = await redis.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    oldest_ts = oldest_entries[0][1]
                    retry_after = max(1, int(window_seconds - (now - oldest_ts)))
                else:
                    retry_after = window_seconds
                return False, retry_after

            # 3. Add current request timestamp
            await redis.zadd(key, {req_id: now})
            await redis.expire(key, window_seconds)

            return True, 0
        except Exception as exc:
            logger.warning("Endpoint rate limit check failed for %s on %s: %s", client_ip, endpoint, exc)
            return True, 0
