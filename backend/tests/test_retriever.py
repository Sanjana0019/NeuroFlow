import asyncio
import json
from uuid import uuid4
import pytest

from pipelines.retrieval.models import ProcessedQuery
from pipelines.retrieval.retriever import Retriever


class FakeRetrieverDBConnection:
    def __init__(self):
        self.fetch_calls = []
        self.sleep_seconds = 0.05

    async def fetch(self, query: str, *args):
        # Simulate slight I/O latency to verify concurrent execution
        await asyncio.sleep(self.sleep_seconds)
        normalized = " ".join(query.split())
        self.fetch_calls.append((normalized, args))

        c_id = uuid4()
        d_id = uuid4()

        if "to_tsvector" in normalized:
            # Sparse FTS
            return [
                {
                    "id": c_id,
                    "document_id": d_id,
                    "content": "Sparse lexical match for query",
                    "chunk_index": 0,
                    "token_count": 10,
                    "metadata": json.dumps({"page_number": 1}),
                    "filename": "sparse_doc.pdf",
                    "score": 0.85,
                }
            ]
        elif "@>" in normalized:
            # Metadata search
            return [
                {
                    "id": c_id,
                    "document_id": d_id,
                    "content": "Metadata match for 2023 climate",
                    "chunk_index": 0,
                    "token_count": 10,
                    "metadata": json.dumps({"year": 2023, "page_number": 2}),
                    "filename": "climate_2023.pdf",
                    "score": 0.92,
                }
            ]
        else:
            # Dense search
            return [
                {
                    "id": c_id,
                    "document_id": d_id,
                    "content": "Dense semantic match for query",
                    "chunk_index": 0,
                    "token_count": 10,
                    "metadata": json.dumps({"page_number": 3}),
                    "filename": "dense_doc.pdf",
                    "score": 0.89,
                }
            ]


class FakeRetrieverDBPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        class AcquireContext:
            def __init__(self, c):
                self.c = c
            async def __aenter__(self):
                return self.c
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return AcquireContext(self.conn)


class FakeEmbedder:
    def __init__(self):
        self.embed_calls = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[0.05] * 1536 for _ in texts]


@pytest.mark.asyncio
async def test_retriever_parallel_execution_and_sources():
    """Retriever runs Dense, Sparse, and Metadata retrieval concurrently via asyncio.gather."""
    conn = FakeRetrieverDBConnection()
    pool = FakeRetrieverDBPool(conn)
    embedder = FakeEmbedder()
    retriever = Retriever(db_pool=pool, embedder=embedder)

    processed = ProcessedQuery(
        original_query="climate impact report",
        expanded_queries=["global temperature rise", "co2 emissions 2023"],
        metadata_filters={"year": 2023, "topic": "climate"},
    )

    import time
    start = time.perf_counter()
    results = await retriever.retrieve_parallel(processed, k=10)
    elapsed = time.perf_counter() - start

    assert "dense" in results
    assert "sparse" in results
    assert "metadata" in results

    assert len(results["dense"]) >= 1
    assert len(results["sparse"]) >= 1
    assert len(results["metadata"]) >= 1

    # Dense result check
    assert results["dense"][0].source == "dense"
    assert "Dense semantic match" in results["dense"][0].content

    # Sparse result check
    assert results["sparse"][0].source == "sparse"
    assert "Sparse lexical match" in results["sparse"][0].content

    # Metadata result check
    assert results["metadata"][0].source == "metadata"
    assert "Metadata match" in results["metadata"][0].content
    assert results["metadata"][0].metadata["year"] == 2023

    # Confirm queries executed concurrently (total time significantly less than sequential sum)
    # 3 dense + 1 sparse + 1 metadata = 5 queries * 0.05s = 0.25s sequential. Concurrent should be ~0.05-0.10s.
    assert elapsed < 0.20
