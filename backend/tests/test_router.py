import json

import pytest

from providers.router import ModelRouter, RoutingCriteria


class FakeRedis:
    def __init__(self, models):
        self.models = models

    async def get(self, key):
        if key != "router:models":
            return None

        return json.dumps(self.models)


@pytest.fixture
def models():
    return [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "supports_vision": True,
            "context_window": 128_000,
            "estimated_latency_ms": 500,
            "estimated_cost_per_call": 0.01,
            "is_fine_tuned": False,
            "is_judge": False,
            "task_types": ["chat"],
        },
        {
            "provider": "openai",
            "model": "gpt-4o",
            "supports_vision": True,
            "context_window": 128_000,
            "estimated_latency_ms": 900,
            "estimated_cost_per_call": 0.05,
            "is_fine_tuned": False,
            "is_judge": True,
            "task_types": ["chat", "evaluation"],
        },
        {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-latest",
            "supports_vision": False,
            "context_window": 200_000,
            "estimated_latency_ms": 600,
            "estimated_cost_per_call": 0.02,
            "is_fine_tuned": False,
            "is_judge": False,
            "task_types": ["chat"],
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini-finetuned",
            "supports_vision": True,
            "context_window": 128_000,
            "estimated_latency_ms": 700,
            "estimated_cost_per_call": 0.03,
            "is_fine_tuned": True,
            "is_judge": False,
            "task_types": ["chat"],
        },
    ]


@pytest.mark.asyncio
async def test_vision_requirement_selects_vision_model(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="chat",
            require_vision=True,
        )
    )

    assert result["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_long_context_requirement(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="chat",
            require_long_context=True,
        )
    )

    assert result["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_max_cost_filters_expensive_models(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="chat",
            max_cost_per_call=0.015,
        )
    )

    assert result["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_latency_budget_filters_slow_models(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="chat",
            latency_budget_ms=550,
        )
    )

    assert result["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_fine_tuned_preference(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="chat",
            prefer_fine_tuned=True,
        )
    )

    assert result["model"] == "gpt-4o-mini-finetuned"


@pytest.mark.asyncio
async def test_evaluation_does_not_select_fine_tuned_model(models):
    router = ModelRouter(FakeRedis(models))

    result = await router.route(
        RoutingCriteria(
            task_type="evaluation",
        )
    )

    assert result["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_no_matching_model_raises_error():
    router = ModelRouter(
        FakeRedis(
            [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "supports_vision": False,
                    "context_window": 128_000,
                    "estimated_latency_ms": 500,
                    "estimated_cost_per_call": 0.01,
                }
            ]
        )
    )

    with pytest.raises(ValueError):
        await router.route(
            RoutingCriteria(
                task_type="chat",
                require_vision=True,
            )
        )