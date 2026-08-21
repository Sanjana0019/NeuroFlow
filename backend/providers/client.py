import time
from typing import AsyncGenerator

from opentelemetry import trace
from redis.asyncio import Redis

from .base import ChatMessage, GenerationResult
from .router import ModelRouter, RoutingCriteria


class NeuroFlowClient:
    """High-level client for model routing, generation, embeddings, and metrics."""

    def __init__(
        self,
        redis: Redis,
        providers: dict[str, object],
    ):
        self.redis = redis
        self.providers = providers
        self.router = ModelRouter(redis)
        self.tracer = trace.get_tracer("neuroflow.providers")

    async def chat(
        self,
        messages: list[ChatMessage],
        routing_criteria: RoutingCriteria,
        **kwargs,
    ) -> GenerationResult:
        """Route a chat request to the appropriate provider."""

        model_config = await self.router.route(routing_criteria)

        model_name = model_config["model"]
        provider_name = model_config["provider"]

        provider = self.providers[provider_name]

        start_time = time.perf_counter()

        with self.tracer.start_as_current_span(
            "llm.chat",
        ) as span:
            result = await provider.complete(
                messages,
                **kwargs,
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the registered embedding provider."""

        provider = self.providers["openai"]

        start_time = time.perf_counter()

        with self.tracer.start_as_current_span(
            "llm.embed",
        ) as span:
            embeddings = await provider.embed(texts)

            latency_ms = (time.perf_counter() - start_time) * 1000

            span.set_attribute("model", "text-embedding-3-small")
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