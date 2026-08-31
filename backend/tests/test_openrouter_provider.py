import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.providers.base import ChatMessage, GenerationResult
from backend.providers.client import NeuroFlowClient
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.providers.router import RoutingCriteria


class FakeRedis:
    def __init__(self, models=None):
        self.calls = {}
        self.costs = {}
        self.models = models or []

    async def get(self, key):
        if key == "router:models":
            return json.dumps(self.models)
        return None

    async def incr(self, key):
        self.calls[key] = self.calls.get(key, 0) + 1
        return self.calls[key]

    async def incrbyfloat(self, key, amount):
        self.costs[key] = self.costs.get(key, 0.0) + amount
        return self.costs[key]

    async def evalsha(self, *args, **kwargs):
        return 1

    async def script_load(self, *args, **kwargs):
        return "fake-sha"

    async def eval(self, *args, **kwargs):
        return 1


def test_openrouter_provider_initialization():
    """Test A: OpenRouter provider initializes with expected base_url and models."""
    provider = OpenRouterProvider(
        api_key="test-sk-or-v1-secretkey",
        model="meta-llama/llama-3.2-3b-instruct:free",
        embedding_model="nvidia/nemotron-3-embed-1b:free",
    )
    assert provider.model == "meta-llama/llama-3.2-3b-instruct:free"
    assert provider.embedding_model == "nvidia/nemotron-3-embed-1b:free"
    assert str(provider.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert provider.cost_per_input_token == 0.0
    assert provider.cost_per_output_token == 0.0


@pytest.mark.asyncio
async def test_provider_registration_and_model_router_generation():
    """Test B & C: Provider registers in NeuroFlowClient and ModelRouter routes generation."""
    models = [
        {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "provider": "openrouter",
            "is_judge": True,
            "is_fine_tuned": False,
            "supports_vision": False,
            "context_window": 128_000,
            "estimated_latency_ms": 250,
            "estimated_cost_per_call": 0.0,
            "task_types": ["generation", "query_processing", "reranking", "evaluation"],
        }
    ]
    fake_redis = FakeRedis(models=models)

    provider_mock = AsyncMock()
    provider_mock.complete.return_value = GenerationResult(
        content="NeuroFlow is an enterprise RAG engine.",
        model="meta-llama/llama-3.2-3b-instruct:free",
        input_tokens=15,
        output_tokens=10,
        latency_ms=120.0,
        cost_usd=0.0,
        finish_reason="stop",
    )

    client = NeuroFlowClient(
        redis=fake_redis,
        providers={"openrouter": provider_mock},
    )

    criteria = RoutingCriteria(task_type="generation")
    messages = [ChatMessage(role="user", content="Explain NeuroFlow")]

    with patch("backend.providers.client.get_global_llm_limiter") as mock_limiter:
        limiter_instance = AsyncMock()
        mock_limiter.return_value = limiter_instance
        result = await client.chat(messages, criteria)
        assert result.content == "NeuroFlow is an enterprise RAG engine."
        assert result.model == "meta-llama/llama-3.2-3b-instruct:free"
        provider_mock.complete.assert_called_once()


@pytest.mark.asyncio
async def test_model_router_embeddings():
    """Test D: NeuroFlowClient.embed calls the configured openrouter provider."""
    fake_redis = FakeRedis()
    provider_mock = AsyncMock()
    fake_vector = [0.01] * 2048
    provider_mock.embed.return_value = [fake_vector]
    provider_mock.embedding_model = "nvidia/nemotron-3-embed-1b:free"

    client = NeuroFlowClient(
        redis=fake_redis,
        providers={"openrouter": provider_mock},
    )

    res = await client.embed(["test document chunk"])
    assert len(res) == 1
    assert len(res[0]) == 2048
    provider_mock.embed.assert_called_once_with(["test document chunk"])


@pytest.mark.asyncio
async def test_missing_api_key_behavior():
    """Test E: When no providers are configured, clean RuntimeError is raised instead of KeyError."""
    fake_redis = FakeRedis()
    client = NeuroFlowClient(
        redis=fake_redis,
        providers={},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.embed(["test text"])
    assert "No LLM/Embedding provider configured" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        await client.chat([ChatMessage(role="user", content="hi")], RoutingCriteria(task_type="generation"))
    assert "No LLM/Embedding provider configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embedding_response_dimension_validation():
    """Test F: Mock OpenRouter embedding response returns exact length."""
    provider = OpenRouterProvider(
        api_key="mock-key",
        embedding_model="nvidia/nemotron-3-embed-1b:free",
    )

    mock_resp = MagicMock()
    item_mock = MagicMock()
    item_mock.index = 0
    item_mock.embedding = [0.05] * 2048
    mock_resp.data = [item_mock]

    with patch.object(provider.client.embeddings, "create", AsyncMock(return_value=mock_resp)):
        embeddings = await provider.embed(["sample chunk"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 2048


def test_no_secret_leakage_in_logs(caplog):
    """Test G: OpenRouter provider initialization and representation never prints API key."""
    raw_secret = "sk-or-v1-secret-key-1234567890abcdef"
    with caplog.at_level(logging.DEBUG):
        provider = OpenRouterProvider(api_key=raw_secret)
        assert raw_secret not in repr(provider)
        assert raw_secret not in str(provider)
        assert raw_secret not in caplog.text
