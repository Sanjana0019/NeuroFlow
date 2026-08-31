import logging
from typing import Any
from backend.providers.openai_provider import OpenAIProvider

logger = logging.getLogger("neuroflow.providers.openrouter")


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider utilizing OpenAI-compatible endpoints with configurable models."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        embedding_model: str = "nvidia/nemotron-3-embed-1b:free",
        base_url: str | None = None,
    ):
        target_base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            base_url=target_base_url,
        )

    @property
    def cost_per_input_token(self) -> float:
        """Free tier models have 0.0 USD cost."""
        if ":free" in self.model:
            return 0.0
        return super().cost_per_input_token

    @property
    def cost_per_output_token(self) -> float:
        """Free tier models have 0.0 USD cost."""
        if ":free" in self.model:
            return 0.0
        return super().cost_per_output_token
