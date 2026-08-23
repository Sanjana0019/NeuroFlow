import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("neuroflow.resilience.circuit_breaker")


class CircuitOpenError(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN or probes exceeded."""
    pass


# Global fallback in-memory state store when Redis is not available
_IN_MEMORY_CIRCUITS: dict[str, dict[str, Any]] = {}
_IN_MEMORY_LOCK = asyncio.Lock()


class CircuitBreaker:
    """Redis-backed distributed Circuit Breaker with CLOSED, OPEN, and HALF_OPEN states."""

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        redis: Any = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.redis = redis

        # Redis key names
        self.state_key = f"circuit:{name}:state"
        self.failure_count_key = f"circuit:{name}:failure_count"
        self.opened_at_key = f"circuit:{name}:opened_at"
        self.half_open_calls_key = f"circuit:{name}:half_open_calls"

    async def _get_state_in_memory(self) -> tuple[str, int, float, int]:
        async with _IN_MEMORY_LOCK:
            if self.name not in _IN_MEMORY_CIRCUITS:
                _IN_MEMORY_CIRCUITS[self.name] = {
                    "state": self.STATE_CLOSED,
                    "failure_count": 0,
                    "opened_at": 0.0,
                    "half_open_calls": 0,
                }
            c = _IN_MEMORY_CIRCUITS[self.name]
            return c["state"], c["failure_count"], c["opened_at"], c["half_open_calls"]

    async def _set_state_in_memory(
        self,
        state: str | None = None,
        failure_count: int | None = None,
        opened_at: float | None = None,
        half_open_calls: int | None = None,
    ) -> None:
        async with _IN_MEMORY_LOCK:
            if self.name not in _IN_MEMORY_CIRCUITS:
                _IN_MEMORY_CIRCUITS[self.name] = {
                    "state": self.STATE_CLOSED,
                    "failure_count": 0,
                    "opened_at": 0.0,
                    "half_open_calls": 0,
                }
            c = _IN_MEMORY_CIRCUITS[self.name]
            if state is not None:
                c["state"] = state
            if failure_count is not None:
                c["failure_count"] = failure_count
            if opened_at is not None:
                c["opened_at"] = opened_at
            if half_open_calls is not None:
                c["half_open_calls"] = half_open_calls

    async def get_state(self) -> tuple[str, int, float, int]:
        """Fetch current circuit state from Redis or fallback store."""
        if not self.redis:
            return await self._get_state_in_memory()

        try:
            state_raw = await self.redis.get(self.state_key)
            if state_raw is None:
                state = self.STATE_CLOSED
            else:
                state = state_raw.decode("utf-8") if isinstance(state_raw, bytes) else str(state_raw)

            fc_raw = await self.redis.get(self.failure_count_key)
            failure_count = int(fc_raw) if fc_raw is not None else 0

            oa_raw = await self.redis.get(self.opened_at_key)
            opened_at = float(oa_raw) if oa_raw is not None else 0.0

            ho_raw = await self.redis.get(self.half_open_calls_key)
            half_open_calls = int(ho_raw) if ho_raw is not None else 0

            return state, failure_count, opened_at, half_open_calls
        except Exception as exc:
            logger.warning("Redis error getting circuit state for '%s': %s", self.name, exc)
            return await self._get_state_in_memory()

    async def before_call(self) -> None:
        """Inspect state before call. Transitions OPEN -> HALF_OPEN if recovery_timeout elapsed, or raises CircuitOpenError."""
        now = time.time()
        state, failure_count, opened_at, half_open_calls = await self.get_state()

        if state == self.STATE_OPEN:
            if now - opened_at >= self.recovery_timeout:
                # Transition OPEN -> HALF_OPEN
                logger.info(
                    "Circuit '%s' recovery timeout (%ds) elapsed. Transitioning OPEN -> HALF_OPEN",
                    self.name,
                    self.recovery_timeout,
                )
                if self.redis:
                    try:
                        await self.redis.set(self.state_key, self.STATE_HALF_OPEN)
                        await self.redis.set(self.half_open_calls_key, 1)
                    except Exception:
                        await self._set_state_in_memory(state=self.STATE_HALF_OPEN, half_open_calls=1)
                else:
                    await self._set_state_in_memory(state=self.STATE_HALF_OPEN, half_open_calls=1)
                return
            else:
                remaining = int(self.recovery_timeout - (now - opened_at))
                raise CircuitOpenError(f"Circuit '{self.name}' is OPEN (recovering in {remaining}s)")

        elif state == self.STATE_HALF_OPEN:
            if half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is HALF_OPEN and reached max probe calls ({self.half_open_max_calls})"
                )
            # Increment half open probe count
            if self.redis:
                try:
                    await self.redis.incr(self.half_open_calls_key)
                except Exception:
                    await self._set_state_in_memory(half_open_calls=half_open_calls + 1)
            else:
                await self._set_state_in_memory(half_open_calls=half_open_calls + 1)

    async def record_success(self) -> None:
        """Record successful call. If HALF_OPEN -> closes circuit and resets failures."""
        state, _, _, _ = await self.get_state()

        if state in (self.STATE_HALF_OPEN, self.STATE_CLOSED):
            if state == self.STATE_HALF_OPEN:
                logger.info("Probe call succeeded in HALF_OPEN. Transitioning circuit '%s' -> CLOSED", self.name)
            if self.redis:
                try:
                    await self.redis.set(self.state_key, self.STATE_CLOSED)
                    await self.redis.set(self.failure_count_key, 0)
                    await self.redis.set(self.half_open_calls_key, 0)
                except Exception:
                    await self._set_state_in_memory(state=self.STATE_CLOSED, failure_count=0, half_open_calls=0)
            else:
                await self._set_state_in_memory(state=self.STATE_CLOSED, failure_count=0, half_open_calls=0)

    async def record_failure(self) -> None:
        """Record failed call. Increments failure count; if >= threshold or in HALF_OPEN -> opens circuit."""
        now = time.time()
        state, failure_count, _, _ = await self.get_state()

        if state == self.STATE_HALF_OPEN:
            logger.warning("Probe call failed in HALF_OPEN. Re-opening circuit '%s'", self.name)
            if self.redis:
                try:
                    await self.redis.set(self.state_key, self.STATE_OPEN)
                    await self.redis.set(self.opened_at_key, str(now))
                    await self.redis.set(self.half_open_calls_key, 0)
                except Exception:
                    await self._set_state_in_memory(state=self.STATE_OPEN, opened_at=now, half_open_calls=0)
            else:
                await self._set_state_in_memory(state=self.STATE_OPEN, opened_at=now, half_open_calls=0)

        elif state == self.STATE_CLOSED:
            new_fc = failure_count + 1
            if new_fc >= self.failure_threshold:
                logger.error(
                    "Circuit '%s' reached failure threshold (%d/%d consecutive failures). Transitioning -> OPEN",
                    self.name,
                    new_fc,
                    self.failure_threshold,
                )
                if self.redis:
                    try:
                        await self.redis.set(self.state_key, self.STATE_OPEN)
                        await self.redis.set(self.failure_count_key, new_fc)
                        await self.redis.set(self.opened_at_key, str(now))
                    except Exception:
                        await self._set_state_in_memory(state=self.STATE_OPEN, failure_count=new_fc, opened_at=now)
                else:
                    await self._set_state_in_memory(state=self.STATE_OPEN, failure_count=new_fc, opened_at=now)
            else:
                if self.redis:
                    try:
                        await self.redis.incr(self.failure_count_key)
                    except Exception:
                        await self._set_state_in_memory(failure_count=new_fc)
                else:
                    await self._set_state_in_memory(failure_count=new_fc)

    async def __aenter__(self):
        await self.before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if not issubclass(exc_type, CircuitOpenError):
                await self.record_failure()
            return False
        else:
            await self.record_success()
            return False


_CIRCUIT_REGISTRY: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 3,
    redis: Any = None,
) -> CircuitBreaker:
    """Get or create singleton CircuitBreaker instance for a given name."""
    if name not in _CIRCUIT_REGISTRY or (_CIRCUIT_REGISTRY[name].redis is None and redis is not None):
        _CIRCUIT_REGISTRY[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            redis=redis,
        )
    return _CIRCUIT_REGISTRY[name]


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 3,
    redis: Any = None,
) -> CircuitBreaker:
    """Helper for async with circuit_breaker('openai'): context manager usage."""
    return get_circuit_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        redis=redis,
    )
