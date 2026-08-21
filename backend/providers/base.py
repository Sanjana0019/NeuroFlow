from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class ChatMessage:
    """A message sent to an LLM."""

    role: str
    content: str | list


@dataclass
class GenerationResult:
    """Standardized result returned by an LLM provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str


class BaseLLMProvider(ABC):
    """Common interface that every LLM provider must implement."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate a response progressively."""
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for text inputs."""
        ...

    @property
    @abstractmethod
    def cost_per_input_token(self) -> float:
        """Cost per input token in USD."""
        ...

    @property
    @abstractmethod
    def cost_per_output_token(self) -> float:
        """Cost per output token in USD."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window supported by the provider/model."""
        ...