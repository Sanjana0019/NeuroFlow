from typing import Any

from backend.providers.base import BaseLLMProvider
from backend.providers.client import NeuroFlowClient, get_client
from pipelines.ingestion.models import Chunk


class ChunkEmbedder:
    """Generates embeddings for document chunks using the configured NeuroFlow provider client."""

    EXPECTED_DIMENSION = 2048

    def __init__(
        self,
        client: NeuroFlowClient | BaseLLMProvider | None = None,
        expected_dimension: int = EXPECTED_DIMENSION,
        validate_dimension: bool = True,
    ) -> None:
        """Initialize embedder with an optional client and dimension validation."""
        self.client = client
        self.expected_dimension = expected_dimension
        self.validate_dimension = validate_dimension

    def _resolve_client(self) -> Any:
        """Retrieve provided client or fallback to the application singleton."""
        if self.client is not None:
            return self.client
        return get_client()

    async def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Generate and attach vector embeddings to a list of Chunk objects in-place."""
        if not chunks:
            return []

        client = self._resolve_client()
        texts = [chunk.content for chunk in chunks]

        embeddings = await client.embed(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: expected {len(chunks)} embeddings, "
                f"got {len(embeddings)}"
            )

        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            if self.validate_dimension and self.expected_dimension is not None:
                if len(vector) != self.expected_dimension:
                    raise ValueError(
                        f"Chunk at index {index} received embedding dimension {len(vector)}, "
                        f"expected {self.expected_dimension}"
                    )
            chunk.embedding = vector

        return chunks
