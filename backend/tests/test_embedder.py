import pytest

from backend.providers.client import set_client
from pipelines.ingestion.embedder import ChunkEmbedder
from pipelines.ingestion.models import Chunk


class FakeEmbeddingProvider:
    """Mock embedding provider returning deterministic mock vectors."""

    def __init__(self, dimension: int = 1536, should_fail: bool = False):
        self.dimension = dimension
        self.should_fail = should_fail
        self.recorded_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.should_fail:
            raise RuntimeError("Upstream provider embedding service error")

        self.recorded_calls.append(texts)
        # Return unique vector per text based on index
        return [
            [float(i + 1) / 100.0] * self.dimension
            for i in range(len(texts))
        ]


@pytest.mark.asyncio
async def test_single_chunk_embedding():
    """A single chunk can be embedded and has the vector attached."""
    provider = FakeEmbeddingProvider(dimension=1536)
    embedder = ChunkEmbedder(client=provider)

    chunk = Chunk(
        content="NeuroFlow embedding test content",
        chunk_index=0,
        token_count=5,
        page_number=1,
        metadata={"source": "test.txt"},
    )

    result = await embedder.embed_chunks([chunk])

    assert len(result) == 1
    assert result[0].embedding is not None
    assert len(result[0].embedding) == 1536
    assert result[0].embedding[0] == 0.01
    assert provider.recorded_calls == [["NeuroFlow embedding test content"]]


@pytest.mark.asyncio
async def test_multiple_chunks_preserve_order():
    """Multiple chunks preserve their order and receive corresponding embeddings."""
    provider = FakeEmbeddingProvider(dimension=1536)
    embedder = ChunkEmbedder(client=provider)

    chunks = [
        Chunk(content=f"Chunk content {i}", chunk_index=i, token_count=4, page_number=1)
        for i in range(5)
    ]

    result = await embedder.embed_chunks(chunks)

    assert len(result) == 5
    for i, chunk in enumerate(result):
        assert chunk.chunk_index == i
        assert chunk.content == f"Chunk content {i}"
        assert chunk.embedding is not None
        assert chunk.embedding[0] == pytest.approx(float(i + 1) / 100.0)

    assert provider.recorded_calls == [
        [f"Chunk content {i}" for i in range(5)]
    ]


@pytest.mark.asyncio
async def test_empty_chunk_list_handled_sensibly():
    """Passing an empty list of chunks returns empty without calling provider."""
    provider = FakeEmbeddingProvider()
    embedder = ChunkEmbedder(client=provider)

    result = await embedder.embed_chunks([])

    assert result == []
    assert len(provider.recorded_calls) == 0


@pytest.mark.asyncio
async def test_dimension_validation_mismatch():
    """Mismatched embedding dimensions raise ValueError."""
    provider = FakeEmbeddingProvider(dimension=512)
    embedder = ChunkEmbedder(client=provider, expected_dimension=1536, validate_dimension=True)

    chunk = Chunk(content="Test", chunk_index=0, token_count=1)

    with pytest.raises(ValueError, match="received embedding dimension 512, expected 1536"):
        await embedder.embed_chunks([chunk])


@pytest.mark.asyncio
async def test_count_mismatch_raises_error():
    """If provider returns fewer or more embeddings than chunks, raise ValueError."""
    class BadProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1536]  # Always returns 1 embedding regardless of input length

    embedder = ChunkEmbedder(client=BadProvider())
    chunks = [
        Chunk(content="Text 1", chunk_index=0, token_count=2),
        Chunk(content="Text 2", chunk_index=1, token_count=2),
    ]

    with pytest.raises(ValueError, match="Embedding count mismatch"):
        await embedder.embed_chunks(chunks)


@pytest.mark.asyncio
async def test_provider_errors_are_surfaced():
    """Errors raised by the provider are surfaced clearly without suppression."""
    provider = FakeEmbeddingProvider(should_fail=True)
    embedder = ChunkEmbedder(client=provider)

    chunk = Chunk(content="Test", chunk_index=0, token_count=1)

    with pytest.raises(RuntimeError, match="Upstream provider embedding service error"):
        await embedder.embed_chunks([chunk])


@pytest.mark.asyncio
async def test_singleton_client_resolution():
    """When no client is passed to constructor, it resolves via get_client()."""
    provider = FakeEmbeddingProvider(dimension=1536)
    set_client(provider)

    embedder = ChunkEmbedder()
    chunk = Chunk(content="Singleton test", chunk_index=0, token_count=2)

    result = await embedder.embed_chunks([chunk])

    assert len(result) == 1
    assert result[0].embedding is not None
    assert len(result[0].embedding) == 1536
