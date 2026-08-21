import httpx
import pytest

from openai import RateLimitError

from providers.base import ChatMessage
from providers.openai_provider import OpenAIProvider


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50


class FakeMessage:
    content = "Hello from OpenAI"


class FakeChoice:
    message = FakeMessage()
    finish_reason = "stop"


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeStreamChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [FakeStreamChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        if kwargs.get("stream"):
            async def generate():
                yield FakeStreamChunk("Hello ")
                yield FakeStreamChunk("world")

            return generate()

        return FakeResponse()


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Item:
            def __init__(self, index):
                self.index = index
                self.embedding = [float(index)]

        class Response:
            data = [
                Item(index)
                for index in reversed(range(len(kwargs["input"])))
            ]

        return Response()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def provider():
    provider = OpenAIProvider(api_key="fake-key")
    provider.client = FakeClient()
    return provider


@pytest.mark.asyncio
async def test_complete_returns_generation_result(provider):
    result = await provider.complete(
        [
            ChatMessage(
                role="user",
                content="Hello",
            )
        ]
    )

    assert result.content == "Hello from OpenAI"
    assert result.model == "gpt-4o-mini"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.finish_reason == "stop"

    expected_cost = (
        100 * (0.15 / 1_000_000)
        + 50 * (0.60 / 1_000_000)
    )

    assert result.cost_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_stream_yields_progressive_text(provider):
    chunks = []

    async for chunk in provider.stream(
        [
            ChatMessage(
                role="user",
                content="Hello",
            )
        ]
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_embeddings_are_batched_at_100(provider):
    texts = [f"text-{index}" for index in range(205)]

    embeddings = await provider.embed(texts)

    calls = provider.client.embeddings.calls

    assert len(calls) == 3
    assert len(calls[0]["input"]) == 100
    assert len(calls[1]["input"]) == 100
    assert len(calls[2]["input"]) == 5
    assert len(embeddings) == 205


def test_openai_pricing(provider):
    assert provider.cost_per_input_token == pytest.approx(
        0.15 / 1_000_000
    )

    assert provider.cost_per_output_token == pytest.approx(
        0.60 / 1_000_000
    )


@pytest.mark.asyncio
async def test_complete_retries_after_rate_limit(monkeypatch, provider):
    attempts = 0

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            request = httpx.Request(
                "POST",
                "https://api.openai.com/v1/chat/completions",
            )

            response = httpx.Response(
                429,
                request=request,
            )

            raise RateLimitError(
                message="rate limited",
                response=response,
                body=None,
            )

        return FakeResponse()

    provider.client.chat.completions.create = fake_create

    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    monkeypatch.setattr(
        "providers.openai_provider.asyncio.sleep",
        fake_sleep,
    )

    result = await provider.complete(
        [
            ChatMessage(
                role="user",
                content="Hello",
            )
        ]
    )

    assert attempts == 3
    assert result.content == "Hello from OpenAI"
    assert sleep_delays == [1, 2]