import asyncio
import time

from openai import AsyncOpenAI, RateLimitError

from providers.base import (
    BaseLLMProvider,
    ChatMessage,
    GenerationResult,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible LLM provider."""

    PRICING = {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60,
        },
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ):
        self.model = model
        self.embedding_model = embedding_model

        client_kwargs = {"api_key": api_key}

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

    @property
    def cost_per_input_token(self) -> float:
        """USD cost per input token."""
        return self.PRICING.get(
            self.model,
            {"input": 0.0, "output": 0.0},
        )["input"] / 1_000_000

    @property
    def cost_per_output_token(self) -> float:
        """USD cost per output token."""
        return self.PRICING.get(
            self.model,
            {"input": 0.0, "output": 0.0},
        )["output"] / 1_000_000

    @property
    def context_window(self) -> int:
        """Context window used by the router."""
        context_windows = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
        }

        return context_windows.get(self.model, 32_000)

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> dict:
        """Convert our common message type to OpenAI's format."""
        return {
            "role": message.role,
            "content": message.content,
        }

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        """Generate a complete response with rate-limit retries."""

        max_retries = 3

        for attempt in range(max_retries + 1):
            started_at = time.perf_counter()

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        self._message_to_dict(message)
                        for message in messages
                    ],
                    **kwargs,
                )

                latency_ms = (
                    time.perf_counter() - started_at
                ) * 1000

                choice = response.choices[0]
                usage = response.usage

                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return GenerationResult(
                    content=choice.message.content or "",
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=self._calculate_cost(
                        input_tokens,
                        output_tokens,
                    ),
                    finish_reason=choice.finish_reason or "unknown",
                )

            except RateLimitError as exc:
                if attempt >= max_retries:
                    raise

                retry_after = getattr(
                    exc,
                    "retry_after",
                    None,
                )

                if retry_after is None:
                    retry_after = 2 ** attempt

                await asyncio.sleep(retry_after)

        raise RuntimeError("Unreachable retry state.")

    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ):
        """Stream generated text progressively."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                self._message_to_dict(message)
                for message in messages
            ],
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if not chunk.choices:
                continue

            token = chunk.choices[0].delta.content

            if token:
                yield token

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings in batches of 100."""

        embeddings: list[list[float]] = []
        batch_size = 100

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float",
            )

            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            embeddings.extend(
                item.embedding
                for item in ordered
            )

        return embeddings