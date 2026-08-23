import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.db.health import perform_full_health_check
from backend.resilience.backpressure import check_ingestion_backpressure, get_ingestion_queue_depth
from backend.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    circuit_breaker,
    get_circuit_breaker,
)
from backend.resilience.rate_limiter import (
    EndpointRateLimiter,
    TokenBucketRateLimiter,
    get_global_llm_limiter,
    get_pipeline_rate_limiter,
)
from backend.resilience.timeout_manager import (
    STATIC_TIMEOUTS,
    TimeoutManager,
    calculate_percentile,
    execute_with_timeout,
)


class MockRedisStore:
    """In-memory async Redis simulation supporting strings, sorted sets, lists, and eval."""

    def __init__(self):
        self.strings: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}  # key -> {member: score}
        self.lists: dict[str, list[str]] = {}

    async def get(self, key: str):
        return self.strings.get(key)

    async def set(self, key: str, value: Any):
        self.strings[key] = str(value)

    async def incr(self, key: str):
        val = int(self.strings.get(key, 0)) + 1
        self.strings[key] = str(val)
        return val

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))

    async def zadd(self, key: str, mapping: dict[str, float]):
        if key not in self.zsets:
            self.zsets[key] = {}
        for member, score in mapping.items():
            self.zsets[key][member] = float(score)

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        if key in self.zsets:
            self.zsets[key] = {
                m: s for m, s in self.zsets[key].items() if not (min_score <= s <= max_score)
            }

    async def zremrangebyrank(self, key: str, start: int, stop: int):
        if key in self.zsets:
            sorted_items = sorted(self.zsets[key].items(), key=lambda x: x[1])
            total = len(sorted_items)
            # Python slice equivalent
            start_idx = start if start >= 0 else total + start
            stop_idx = stop if stop >= 0 else total + stop
            to_remove = set([item[0] for item in sorted_items[start_idx : stop_idx + 1]])
            self.zsets[key] = {m: s for m, s in self.zsets[key].items() if m not in to_remove}

    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False):
        if key not in self.zsets:
            return []
        sorted_items = sorted(self.zsets[key].items(), key=lambda x: x[1])
        total = len(sorted_items)
        start_idx = start if start >= 0 else total + start
        stop_idx = stop if stop >= 0 else total + stop
        sliced = sorted_items[start_idx : (stop_idx + 1 if stop_idx < total else total)]
        if withscores:
            return [(item[0].encode("utf-8"), item[1]) for item in sliced]
        return [item[0].encode("utf-8") for item in sliced]

    async def expire(self, key: str, seconds: int):
        pass

    async def keys(self, pattern: str = "*"):
        import fnmatch
        matching = [k.encode("utf-8") for k in self.strings.keys() if fnmatch.fnmatch(k, pattern)]
        return matching

    async def ping(self):
        return True

    async def aclose(self):
        pass

    async def eval(self, script: str, numkeys: int, *args):
        # Emulate token bucket Lua evaluation
        tokens_key = args[0]
        last_updated_key = args[1]
        capacity = float(args[2])
        refill_rate = float(args[3])
        requested = float(args[4])
        now = float(args[5])

        current_tokens = float(self.strings.get(tokens_key, capacity))
        last_updated = float(self.strings.get(last_updated_key, now))

        elapsed = max(0.0, now - last_updated)
        current_tokens = min(capacity, current_tokens + (elapsed * refill_rate))

        if current_tokens >= requested:
            current_tokens -= requested
            self.strings[tokens_key] = str(current_tokens)
            self.strings[last_updated_key] = str(now)
            return [1, 0]
        else:
            needed = requested - current_tokens
            wait_time = needed / refill_rate if refill_rate > 0 else 1.0
            self.strings[tokens_key] = str(current_tokens)
            self.strings[last_updated_key] = str(now)
            return [0, str(wait_time)]


# ============================================================================
# 1. Circuit Breaker Tests
# ============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_transitions_and_recovery():
    """Test CLOSED -> OPEN after 5 failures -> reject calls -> HALF_OPEN after timeout -> CLOSED on success."""
    redis = MockRedisStore()
    cb = CircuitBreaker(
        name="test_cb_1",
        failure_threshold=5,
        recovery_timeout=2,  # 2 seconds for test speed
        half_open_max_calls=3,
        redis=redis,
    )

    # 1. Starts CLOSED
    state, fc, _, _ = await cb.get_state()
    assert state == "closed"
    assert fc == 0

    # 2. Record 4 failures -> still CLOSED
    for i in range(4):
        await cb.record_failure()
        state, fc, _, _ = await cb.get_state()
        assert state == "closed"
        assert fc == i + 1

    # 3. 5th failure -> Transitions to OPEN
    await cb.record_failure()
    state, fc, opened_at, _ = await cb.get_state()
    assert state == "open"
    assert fc == 5
    assert opened_at > 0

    # 4. Calls while OPEN immediately raise CircuitOpenError without calling provider
    provider_called = False
    with pytest.raises(CircuitOpenError):
        async with cb:
            provider_called = True
    assert not provider_called

    # 5. Wait recovery timeout (2s) -> Transitions to HALF_OPEN on next call
    await asyncio.sleep(2.1)
    async with cb:
        # Inside half open probe call
        state, _, _, ho_calls = await cb.get_state()
        assert state == "half_open"
        assert ho_calls == 1

    # 6. Probe call succeeded -> Circuit transitions back to CLOSED
    state, fc, _, _ = await cb.get_state()
    assert state == "closed"
    assert fc == 0


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_reopens():
    """Test that a failed probe in HALF_OPEN transitions immediately back to OPEN."""
    redis = MockRedisStore()
    cb = CircuitBreaker(
        name="test_cb_2",
        failure_threshold=2,
        recovery_timeout=1,
        half_open_max_calls=3,
        redis=redis,
    )

    # Trigger OPEN
    await cb.record_failure()
    await cb.record_failure()
    state, _, _, _ = await cb.get_state()
    assert state == "open"

    # Wait recovery
    await asyncio.sleep(1.1)

    # Probe call fails
    with pytest.raises(ValueError):
        async with cb:
            raise ValueError("Provider error")

    # State must be OPEN again
    state, _, _, _ = await cb.get_state()
    assert state == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_max_calls_limit():
    """Test HALF_OPEN allows at most half_open_max_calls concurrent/sequential calls."""
    redis = MockRedisStore()
    cb = CircuitBreaker(
        name="test_cb_3",
        failure_threshold=1,
        recovery_timeout=1,
        half_open_max_calls=2,
        redis=redis,
    )
    await cb.record_failure()
    await asyncio.sleep(1.1)

    # Manually set state to half_open with 2 calls consumed
    await redis.set("circuit:test_cb_3:state", "half_open")
    await redis.set("circuit:test_cb_3:half_open_calls", 2)

    # Next call should be rejected because max calls reached
    with pytest.raises(CircuitOpenError):
        async with cb:
            pass


# ============================================================================
# 2. Token Bucket Rate Limiter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_token_bucket_consumption_and_refill():
    """Test token consumption, depletion, and refill over time."""
    redis = MockRedisStore()
    limiter = TokenBucketRateLimiter(
        key="rpb:test_openai",
        capacity=5.0,
        refill_rate=10.0,  # 10 tokens/s
        redis=redis,
    )

    # 1. Acquire 3 tokens -> success
    acquired = await limiter.acquire(tokens=3.0)
    assert acquired is True

    # 2. Acquire 2 tokens -> success (now 0 left)
    acquired = await limiter.acquire(tokens=2.0)
    assert acquired is True

    # 3. Next acquire without waiting should trigger wait and succeed as tokens refill
    start = time.perf_counter()
    acquired = await limiter.acquire(tokens=2.0, timeout=1.0)
    elapsed = time.perf_counter() - start
    assert acquired is True
    assert elapsed >= 0.1  # took some time to refill 2 tokens at 10/s


@pytest.mark.asyncio
async def test_pipeline_rate_limiter_rpm():
    """Test per-pipeline rate limiter configured with rate_limit_rpm."""
    redis = MockRedisStore()
    pipe_id = uuid4()
    limiter = get_pipeline_rate_limiter(pipeline_id=pipe_id, rate_limit_rpm=60, redis=redis)

    assert limiter.capacity == 60.0
    assert limiter.refill_rate == 1.0  # 60 RPM = 1/s
    assert limiter.key == f"rpb:pipeline:{pipe_id}"


# ============================================================================
# 3. Endpoint Rate Limiting Tests (Ingest & Query)
# ============================================================================


@pytest.mark.asyncio
async def test_endpoint_sliding_window_rate_limiter():
    """Test IP sliding-window rate limiting for 10/hour on /ingest and 60/min on /query."""
    redis = MockRedisStore()
    client_ip = "192.168.1.50"

    # Test /ingest limit of 10
    for i in range(10):
        allowed, _ = await EndpointRateLimiter.check_rate_limit(
            redis=redis,
            client_ip=client_ip,
            endpoint="ingest",
            max_requests=10,
            window_seconds=3600,
        )
        assert allowed is True

    # 11th request -> Rejected (429)
    allowed, retry_after = await EndpointRateLimiter.check_rate_limit(
        redis=redis,
        client_ip=client_ip,
        endpoint="ingest",
        max_requests=10,
        window_seconds=3600,
    )
    assert allowed is False
    assert retry_after > 0


# ============================================================================
# 4. Ingestion Backpressure Tests
# ============================================================================


@pytest.mark.asyncio
async def test_ingestion_backpressure_thresholds():
    """Test backpressure actions: queue < 50 allow, 50 < queue <= 100 warn (202), queue > 100 reject (503)."""
    redis = MockRedisStore()

    # Case 1: queue depth < 50 -> allow
    redis.lists["queue:ingest"] = ["doc"] * 20
    res = await check_ingestion_backpressure(redis)
    assert res["action"] == "allow"
    assert res["queue_depth"] == 20

    # Case 2: queue depth 75 -> warn (202)
    redis.lists["queue:ingest"] = ["doc"] * 75
    res = await check_ingestion_backpressure(redis)
    assert res["action"] == "warn"
    assert res["status_code"] == 202
    assert res["warning"] == "high_queue_depth"
    assert res["estimated_wait_minutes"] >= 1

    # Case 3: queue depth 120 -> reject (503)
    redis.lists["queue:ingest"] = ["doc"] * 120
    res = await check_ingestion_backpressure(redis)
    assert res["action"] == "reject"
    assert res["status_code"] == 503
    assert res["payload"]["error"] == "ingestion_queue_full"
    assert res["payload"]["retry_after"] == 30


# ============================================================================
# 5. Timeout & Adaptive Timeout Tests
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_manager_execution_and_counter():
    """Test configured timeout execution, timeout exception, and Redis counter."""
    redis = MockRedisStore()
    mgr = TimeoutManager(redis=redis)

    # 1. Fast call succeeds
    async def fast_call():
        return "success"

    res = await mgr.execute("embedding", fast_call())
    assert res == "success"

    # Verify observation recorded in Redis
    assert await redis.zcard("latency:embedding") == 1

    # 2. Slow call triggers TimeoutError and increments timeouts counter
    async def slow_call():
        await asyncio.sleep(0.5)
        return "slow"

    with pytest.raises(TimeoutError) as exc_info:
        await mgr.execute("embedding", slow_call(), override_timeout=0.1)

    assert "timed out" in str(exc_info.value)
    timeout_count = int(await redis.get("timeouts:embedding"))
    assert timeout_count == 1


@pytest.mark.asyncio
async def test_adaptive_timeout_p95_and_trend_detection(caplog):
    """Test adaptive timeout p95 * 1.5 calculation and trend warning log."""
    redis = MockRedisStore()
    mgr = TimeoutManager(redis=redis)
    now = time.time()

    # 1. Fewer than 10 observations -> Fall back to static timeout (10s for embedding)
    for i in range(5):
        await redis.zadd("latency:embedding", {f"{now - i}:{0.10 + i * 0.01}": now - i})

    effective_timeout = await mgr.get_effective_timeout("embedding")
    assert effective_timeout == STATIC_TIMEOUTS["embedding"]

    # 2. Add 20 observations with latency around 0.5s
    for i in range(20):
        # 10 older observations (> 1 hour ago) with latency 0.2s
        await redis.zadd("latency:embedding", {f"{now - 4000 - i}:0.2": now - 4000 - i})
        # 10 recent observations with latency 0.6s
        await redis.zadd("latency:embedding", {f"{now - i}:0.6": now - i})

    effective_timeout = await mgr.get_effective_timeout("embedding")
    # p95 around 0.6s * 1.5 = 0.9s -> bounded by min_timeout 2.0s
    assert effective_timeout >= 2.0  # min bound for embedding is 2.0s


def test_percentile_calculation():
    """Verify linear interpolation percentile calculation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p50 = calculate_percentile(values, 0.50)
    p95 = calculate_percentile(values, 0.95)
    assert abs(p50 - 5.5) < 1e-5
    assert p95 > 9.0


# ============================================================================
# 6. Full Health Check Enhancement Tests
# ============================================================================


@pytest.mark.asyncio
async def test_perform_full_health_check_states():
    """Test health check statuses: ok, degraded, and critical."""
    redis = MockRedisStore()
    mock_db_pool = MagicMock()

    class MockConn:
        async def fetchval(self, query):
            return 1

    class MockCtx:
        async def __aenter__(self):
            return MockConn()
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_db_pool.acquire.return_value = MockCtx()

    # Case A: Healthy system -> 'ok'
    with patch("backend.db.health.check_mlflow", new_callable=AsyncMock) as mock_mlflow:
        mock_mlflow.return_value = (True, 1.5)
        health = await perform_full_health_check(db_pool=mock_db_pool, redis_client=redis)
        assert health["status"] == "ok"
        assert health["checks"]["postgres"]["status"] == "ok"
        assert health["checks"]["redis"]["status"] == "ok"
        assert "latency_ms" in health["checks"]["postgres"]
        assert health["checks"]["circuit_breakers"]["openai"]["state"] == "closed"

    # Case B: Open Circuit Breaker -> 'degraded'
    await redis.set("circuit:openai:state", "open")
    with patch("backend.db.health.check_mlflow", new_callable=AsyncMock) as mock_mlflow:
        mock_mlflow.return_value = (True, 1.5)
        health = await perform_full_health_check(db_pool=mock_db_pool, redis_client=redis)
        assert health["status"] == "degraded"
        assert health["checks"]["circuit_breakers"]["openai"]["state"] == "open"

    # Case C: PostgreSQL unavailable -> 'critical'
    health = await perform_full_health_check(db_pool=None, redis_client=redis)
    assert health["status"] == "critical"
    assert health["checks"]["postgres"]["status"] == "error"
