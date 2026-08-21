import json
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass
class RoutingCriteria:
    task_type: str
    max_cost_per_call: float | None = None
    require_vision: bool = False
    require_long_context: bool = False
    latency_budget_ms: int | None = None
    prefer_fine_tuned: bool = False


class ModelRouter:
    """Selects the most appropriate registered model for a request."""

    def __init__(self, redis: Redis):
        self.redis = redis

    async def _get_models(self) -> list[dict]:
        """Load registered model configurations from Redis."""
        raw = await self.redis.get("router:models")

        if not raw:
            return []

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        return json.loads(raw)

    async def route(self, criteria: RoutingCriteria) -> dict:
        """Return the best model configuration for the requested criteria."""

        models = await self._get_models()

        # 1. Evaluation must never use a fine-tuned model.
        if criteria.task_type == "evaluation":
            models = [
                model
                for model in models
                if model.get("is_judge", False)
                and not model.get("is_fine_tuned", False)
            ]

        # 2. Vision requirement.
        if criteria.require_vision:
            models = [
                model
                for model in models
                if model.get("supports_vision", False)
            ]

        # 3. Long-context requirement.
        if criteria.require_long_context:
            models = [
                model
                for model in models
                if model.get("context_window", 0) > 100_000
            ]

        # 4. Latency requirement.
        if criteria.latency_budget_ms is not None:
            models = [
                model
                for model in models
                if model.get("estimated_latency_ms", float("inf"))
                <= criteria.latency_budget_ms
            ]

        # 5. Prefer a fine-tuned model when one exists for this task.
        if criteria.prefer_fine_tuned:
            fine_tuned = [
                model
                for model in models
                if model.get("is_fine_tuned", False)
                and criteria.task_type in model.get("task_types", [])
            ]

            if fine_tuned:
                models = fine_tuned

        # 6. Cost constraint.
        if criteria.max_cost_per_call is not None:
            models = [
                model
                for model in models
                if model.get("estimated_cost_per_call", float("inf"))
                <= criteria.max_cost_per_call
            ]

        if not models:
            raise ValueError(
                f"No registered model satisfies routing criteria: {criteria}"
            )

        # Default: cheapest model satisfying all hard constraints.
        return min(
            models,
            key=lambda model: model.get(
                "estimated_cost_per_call",
                float("inf"),
            ),
        )