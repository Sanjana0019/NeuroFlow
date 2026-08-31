import time
from typing import AsyncGenerator

from opentelemetry import trace
from redis.asyncio import Redis

from resilience.circuit_breaker import circuit_breaker
from resilience.rate_limiter import get_global_llm_limiter
from resilience.timeout_manager import TimeoutManager

from .base import ChatMessage, GenerationResult
from .router import ModelRouter, RoutingCriteria


class NeuroFlowClient:
    """High-level client for model routing, generation, embeddings, and metrics with resilience."""

    def __init__(
        self,
        redis: Redis,
        providers: dict[str, object],
    ):
        self.redis = redis
        self.providers = providers
        self.router = ModelRouter(redis)
        self.tracer = trace.get_tracer("neuroflow.providers")
        self.timeout_manager = TimeoutManager(redis=redis)

    async def chat(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria,
        **kwargs,
    ) -> GenerationResult:
        if not self.providers:
            raise RuntimeError(
                "No LLM/Embedding provider configured. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
            )

        model_config = await self.router.route(routing_criteria)

        model_name = model_config["model"]
        provider_name = model_config["provider"]

        provider = self.providers.get(provider_name)
        if not provider:
            raise RuntimeError(
                f"Routed provider '{provider_name}' is not registered in NeuroFlowClient. Available: {list(self.providers.keys())}"
            )

        # 1. Global LLM rate limiter
        limiter = get_global_llm_limiter(provider=provider_name, redis=self.redis)
        await limiter.acquire(tokens=1.0)

        start_time = time.perf_counter()

        with self.tracer.start_as_current_span(
            "llm.chat",
        ) as span:
            # 2. Circuit breaker protection
            async with circuit_breaker(provider_name, redis=self.redis):
                # 3. Timeout manager with adaptive p95 latency tracking
                result = await self.timeout_manager.execute(
                    "chat_completion",
                    provider.complete(
                        messages,
                        **kwargs,
                    ),
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            span.set_attribute("model", result.model)
            span.set_attribute("input_tokens", result.input_tokens)
            span.set_attribute("output_tokens", result.output_tokens)
            span.set_attribute("cost_usd", result.cost_usd)
            span.set_attribute("latency_ms", latency_ms)

        await self._record_metrics(
            result.model,
            result.cost_usd,
        )

        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria,
        **kwargs,
    ) -> tuple[AsyncGenerator[str, None], str]:
        """Route a chat stream request to the appropriate provider."""
        if not self.providers:
            raise RuntimeError(
                "No LLM/Embedding provider configured. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
            )

        model_config = await self.router.route(routing_criteria)
        model_name = model_config["model"]
        provider_name = model_config["provider"]

        provider = self.providers.get(provider_name)
        if not provider:
            raise RuntimeError(
                f"Routed provider '{provider_name}' is not registered in NeuroFlowClient. Available: {list(self.providers.keys())}"
            )

        # Rate limiter
        limiter = get_global_llm_limiter(provider=provider_name, redis=self.redis)
        await limiter.acquire(tokens=1.0)

        return provider.stream(messages, **kwargs), model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the registered embedding provider with timeout and circuit breaker."""
        if not self.providers:
            raise RuntimeError(
                "No LLM/Embedding provider configured. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
            )

        provider_name = (
            "openrouter"
            if "openrouter" in self.providers
            else ("openai" if "openai" in self.providers else next(iter(self.providers.keys())))
        )
        provider = self.providers[provider_name]
        embedding_model_name = getattr(provider, "embedding_model", "text-embedding-3-small")

        start_time = time.perf_counter()

        with self.tracer.start_as_current_span(
            "llm.embed",
        ) as span:
            async with circuit_breaker(provider_name, redis=self.redis):
                embeddings = await self.timeout_manager.execute(
                    "embedding",
                    provider.embed(texts),
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            span.set_attribute("model", embedding_model_name)
            span.set_attribute("latency_ms", latency_ms)

        return embeddings

    async def _record_metrics(
        self,
        model_name: str,
        cost_usd: float,
    ) -> None:
        """Record model call count and accumulated cost in Redis."""

        calls_key = f"metrics:model:{model_name}:calls"
        cost_key = f"metrics:model:{model_name}:cost_usd"

        await self.redis.incr(calls_key)
        await self.redis.incrbyfloat(cost_key, cost_usd)


_client: NeuroFlowClient | None = None


def get_client() -> NeuroFlowClient:
    """Return the configured NeuroFlow client singleton."""

    if _client is None:
        raise RuntimeError(
            "NeuroFlowClient has not been initialized."
        )

    return _client


def set_client(client: NeuroFlowClient) -> None:
    """Set the application-wide NeuroFlow client."""

    global _client
    _client = client