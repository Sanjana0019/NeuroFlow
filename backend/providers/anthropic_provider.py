import asyncio
import time
from typing import AsyncGenerator

from anthropic import AsyncAnthropic, RateLimitError

from .base import BaseLLMProvider, ChatMessage, GenerationResult


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude implementation of the NeuroFlow LLM provider interface."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-latest",
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    @property
    def cost_per_input_token(self) -> float:
        """Approximate USD cost per input token."""
        prices = {
            "claude-3-5-haiku-latest": 0.80 / 1_000_000,
            "claude-3-5-sonnet-latest": 3.00 / 1_000_000,
        }
        return prices.get(self.model, 0.0)

    @property
    def cost_per_output_token(self) -> float:
        """Approximate USD cost per output token."""
        prices = {
            "claude-3-5-haiku-latest": 4.00 / 1_000_000,
            "claude-3-5-sonnet-latest": 15.00 / 1_000_000,
        }
        return prices.get(self.model, 0.0)

    @property
    def context_window(self) -> int:
        return 200_000

    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        """Generate a complete response using Anthropic Claude."""

        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        api_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role != "system"
        ]

        start_time = time.perf_counter()

        for attempt in range(3):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get("max_tokens", 1024),
                    system="\n\n".join(system_messages) if system_messages else None,
                    messages=api_messages,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                content = "".join(
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                )

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                cost_usd = (
                    input_tokens * self.cost_per_input_token
                    + output_tokens * self.cost_per_output_token
                )

                return GenerationResult(
                    content=content,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    finish_reason=response.stop_reason or "unknown",
                )

            except RateLimitError as exc:
                if attempt == 2:
                    raise

                retry_after = getattr(exc, "retry_after", None)

                if retry_after is None:
                    retry_after = 2**attempt

                await asyncio.sleep(float(retry_after))

        raise RuntimeError("Anthropic request failed after retries.")

    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text progressively from Anthropic."""

        system_messages = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        api_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role != "system"
        ]

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            system="\n\n".join(system_messages) if system_messages else None,
            messages=api_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic does not provide an embedding API."""

        raise NotImplementedError(
            "Anthropic does not provide embeddings. "
            "Use an embedding-capable provider such as OpenAI."
        )