import httpx
import pytest

from anthropic import RateLimitError

from providers.anthropic_provider import AnthropicProvider
from providers.base import ChatMessage


class FakeUsage:
    input_tokens = 120
    output_tokens = 60


class FakeTextBlock:
    text = "Hello from Claude"


class FakeResponse:
    content = [FakeTextBlock()]
    usage = FakeUsage()
    stop_reason = "end_turn"


class FakeStream:
    def __init__(self):
        self.text_stream = self._generate()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def _generate(self):
        yield "Hello "
        yield "from Claude"


class FakeMessages:
    def __init__(self):
        self.create_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeResponse()

    def stream(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeStream()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


@pytest.fixture
def provider():
    provider = AnthropicProvider(
        api_key="fake-key",
    )
    provider.client = FakeClient()
    return provider


@pytest.mark.asyncio
async def test_complete_returns_generation_result(provider):
    result = await provider.complete(
        [
            ChatMessage(
                role="system",
                content="You are helpful.",
            ),
            ChatMessage(
                role="user",
                content="Hello",
            ),
        ]
    )

    assert result.content == "Hello from Claude"
    assert result.model == "claude-3-5-haiku-latest"
    assert result.input_tokens == 120
    assert result.output_tokens == 60
    assert result.finish_reason == "end_turn"

    expected_cost = (
        120 * (0.80 / 1_000_000)
        + 60 * (4.00 / 1_000_000)
    )

    assert result.cost_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_system_messages_are_top_level(provider):
    messages = [
        ChatMessage(
            role="system",
            content="You are helpful.",
        ),
        ChatMessage(
            role="user",
            content="Hello",
        ),
    ]

    await provider.complete(messages)

    call = provider.client.messages.create_calls[0]

    assert call["system"] == "You are helpful."
    assert call["messages"] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


@pytest.mark.asyncio
async def test_stream_yields_progressive_text(provider):
    chunks = []

    async for chunk in provider.stream(
        [
            ChatMessage(
                role="system",
                content="Be concise.",
            ),
            ChatMessage(
                role="user",
                content="Hello",
            ),
        ]
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "from Claude"]

    call = provider.client.messages.create_calls[0]

    assert call["system"] == "Be concise."
    assert call["messages"] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


@pytest.mark.asyncio
async def test_embed_raises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.embed(["Hello"])


def test_anthropic_pricing(provider):
    assert provider.cost_per_input_token == pytest.approx(
        0.80 / 1_000_000
    )

    assert provider.cost_per_output_token == pytest.approx(
        4.00 / 1_000_000
    )


def test_anthropic_context_window(provider):
    assert provider.context_window == 200_000


@pytest.mark.asyncio
async def test_complete_retries_after_rate_limit(monkeypatch, provider):
    attempts = 0

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            request = httpx.Request(
                "POST",
                "https://api.anthropic.com/v1/messages",
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

    provider.client.messages.create = fake_create

    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    monkeypatch.setattr(
        "providers.anthropic_provider.asyncio.sleep",
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
    assert result.content == "Hello from Claude"
    assert sleep_delays == [1, 2]