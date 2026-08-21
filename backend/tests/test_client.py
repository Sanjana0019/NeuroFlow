import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from providers.base import ChatMessage, GenerationResult
from providers.client import (
    NeuroFlowClient,
    get_client,
    set_client,
)
from providers.router import RoutingCriteria


class FakeRedis:
    def __init__(self):
        self.calls = {}
        self.costs = {}

    async def get(self, key):
        if key == "router:models":
            return (
                '[{"provider": "fake", '
                '"model": "fake-model", '
                '"estimated_cost_per_call": 0.01}]'
            )

        return None

    async def incr(self, key):
        self.calls[key] = self.calls.get(key, 0) + 1
        return self.calls[key]

    async def incrbyfloat(self, key, amount):
        self.costs[key] = self.costs.get(key, 0.0) + amount
        return self.costs[key]


class FakeProvider:
    def __init__(self):
        self.complete_calls = []
        self.embed_calls = []

    async def complete(self, messages, **kwargs):
        self.complete_calls.append(
            {
                "messages": messages,
                "kwargs": kwargs,
            }
        )

        return GenerationResult(
            content="Fake response",
            model="fake-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
            cost_usd=0.000045,
            finish_reason="stop",
        )

    async def embed(self, texts):
        self.embed_calls.append(texts)

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


@pytest.mark.asyncio
async def test_chat_routes_to_selected_provider():
    redis = FakeRedis()
    provider = FakeProvider()

    client = NeuroFlowClient(
        redis=redis,
        providers={"fake": provider},
    )

    messages = [
        ChatMessage(
            role="user",
            content="Hello",
        )
    ]

    result = await client.chat(
        messages=messages,
        routing_criteria=RoutingCriteria(
            task_type="chat",
        ),
    )

    assert result.content == "Fake response"
    assert result.model == "fake-model"

    assert len(provider.complete_calls) == 1
    assert provider.complete_calls[0]["messages"] == messages


@pytest.mark.asyncio
async def test_chat_records_redis_metrics():
    redis = FakeRedis()
    provider = FakeProvider()

    client = NeuroFlowClient(
        redis=redis,
        providers={"fake": provider},
    )

    await client.chat(
        messages=[
            ChatMessage(
                role="user",
                content="Hello",
            )
        ],
        routing_criteria=RoutingCriteria(
            task_type="chat",
        ),
    )

    assert redis.calls[
        "metrics:model:fake-model:calls"
    ] == 1

    assert redis.costs[
        "metrics:model:fake-model:cost_usd"
    ] == pytest.approx(0.000045)


@pytest.mark.asyncio
async def test_embed_uses_openai_provider():
    redis = FakeRedis()
    provider = FakeProvider()

    client = NeuroFlowClient(
        redis=redis,
        providers={"openai": provider},
    )

    texts = [
        "Hello",
        "NeuroFlow",
    ]

    embeddings = await client.embed(texts)

    assert embeddings == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]

    assert provider.embed_calls == [texts]


@pytest.mark.asyncio
async def test_chat_creates_otel_span_with_attributes():
    exporter = InMemorySpanExporter()

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(exporter)
    )

    original_provider = trace.get_tracer_provider()

    try:
        trace.set_tracer_provider(tracer_provider)

        redis = FakeRedis()
        provider = FakeProvider()

        client = NeuroFlowClient(
            redis=redis,
            providers={"fake": provider},
        )

        await client.chat(
            messages=[
                ChatMessage(
                    role="user",
                    content="Hello",
                )
            ],
            routing_criteria=RoutingCriteria(
                task_type="chat",
            ),
        )

        spans = exporter.get_finished_spans()

        assert len(spans) == 1

        span = spans[0]

        assert span.name == "llm.chat"
        assert span.attributes["model"] == "fake-model"
        assert span.attributes["input_tokens"] == 100
        assert span.attributes["output_tokens"] == 50
        assert span.attributes["cost_usd"] == pytest.approx(
            0.000045
        )
        assert span.attributes["latency_ms"] >= 0

    finally:
        exporter.clear()

        if isinstance(
            original_provider,
            TracerProvider,
        ):
            trace.set_tracer_provider(original_provider)


def test_client_singleton():
    redis = FakeRedis()
    provider = FakeProvider()

    client = NeuroFlowClient(
        redis=redis,
        providers={"fake": provider},
    )

    set_client(client)

    assert get_client() is client