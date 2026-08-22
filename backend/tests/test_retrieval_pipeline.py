import json
from uuid import uuid4
import pytest

from backend.providers.base import GenerationResult
from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import Reranker
from pipelines.retrieval.retriever import Retriever


class MockPipelineLLMClient:
    async def chat(self, messages, routing_criteria, **kwargs):
        system_content = messages[0].content
        if "query understanding" in system_content:
            # Query Processor response
            payload = {
                "expanded_queries": ["transformer attention mechanism", "self-attention calculation"],
                "metadata_filters": {"year": 2023},
                "query_type": "analytical",
            }
            return GenerationResult(
                content=json.dumps(payload),
                model="gpt-4o-mini",
                input_tokens=30,
                output_tokens=20,
                latency_ms=50.0,
                cost_usd=0.00001,
                finish_reason="stop",
            )
        elif "relevance scoring" in system_content:
            # Reranker response
            user_prompt = messages[-1].content
            if "Multi-Head Attention" in user_prompt:
                score = "9.8"
            else:
                score = "4.0"
            return GenerationResult(
                content=score,
                model="gpt-4o-mini",
                input_tokens=30,
                output_tokens=5,
                latency_ms=50.0,
                cost_usd=0.00001,
                finish_reason="stop",
            )
        return GenerationResult(
            content="7.0",
            model="gpt-4o-mini",
            input_tokens=20,
            output_tokens=5,
            latency_ms=50.0,
            cost_usd=0.00001,
            finish_reason="stop",
        )


class MockPipelineDBConnection:
    async def fetch(self, query: str, *args):
        c1 = uuid4()
        c2 = uuid4()
        d1 = uuid4()
        normalized = " ".join(query.split())

        if "@>" in normalized:
            return [
                {
                    "id": c1,
                    "document_id": d1,
                    "content": "Multi-Head Attention maps queries and keys to attention weights in 2023.",
                    "chunk_index": 0,
                    "token_count": 12,
                    "metadata": json.dumps({"year": 2023, "page_number": 5}),
                    "filename": "transformers_survey.pdf",
                    "score": 0.95,
                }
            ]
        elif "to_tsvector" in normalized:
            return [
                {
                    "id": c2,
                    "document_id": d1,
                    "content": "Attention is a mechanism in deep learning neural networks.",
                    "chunk_index": 1,
                    "token_count": 10,
                    "metadata": json.dumps({"year": 2022, "page_number": 2}),
                    "filename": "deep_learning.pdf",
                    "score": 0.82,
                }
            ]
        else:
            return [
                {
                    "id": c1,
                    "document_id": d1,
                    "content": "Multi-Head Attention maps queries and keys to attention weights in 2023.",
                    "chunk_index": 0,
                    "token_count": 12,
                    "metadata": json.dumps({"year": 2023, "page_number": 5}),
                    "filename": "transformers_survey.pdf",
                    "score": 0.91,
                },
                {
                    "id": c2,
                    "document_id": d1,
                    "content": "Attention is a mechanism in deep learning neural networks.",
                    "chunk_index": 1,
                    "token_count": 10,
                    "metadata": json.dumps({"year": 2022, "page_number": 2}),
                    "filename": "deep_learning.pdf",
                    "score": 0.78,
                },
            ]


class MockPipelineDBPool:
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


class MockPipelineEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.05] * 1536 for _ in texts]


@pytest.mark.asyncio
async def test_full_retrieval_pipeline_flow():
    """Test full pipeline: QueryProcessor -> Retriever -> RRF -> Reranker -> ContextAssembler."""
    llm_client = MockPipelineLLMClient()
    db_conn = MockPipelineDBConnection()
    db_pool = MockPipelineDBPool(db_conn)
    embedder = MockPipelineEmbedder()

    query_processor = QueryProcessor(client=llm_client)
    retriever = Retriever(db_pool=db_pool, embedder=embedder)
    reranker = Reranker(client=llm_client)
    context_assembler = ContextAssembler(token_budget=2000)

    pipeline = RetrievalPipeline(
        retriever=retriever,
        query_processor=query_processor,
        reranker=reranker,
        context_assembler=context_assembler,
    )

    chunks, assembled, processed = await pipeline.run(
        query="How does multi-head attention work in 2023 transformers?",
        mode="full",
        top_k=5,
    )

    # Check ProcessedQuery
    assert processed.query_type == "analytical"
    assert len(processed.expanded_queries) >= 1
    assert processed.metadata_filters.get("year") == 2023

    # Check Reranked Chunks
    assert len(chunks) >= 1
    assert chunks[0].score == 9.8
    assert chunks[0].source == "reranked"
    assert "Multi-Head Attention" in chunks[0].content

    # Check Assembled Context
    assert "[Source 1 — transformers_survey.pdf, page 5]" in assembled.context
    assert assembled.total_tokens > 0
    assert len(assembled.sources) >= 1
    assert assembled.sources[0]["filename"] == "transformers_survey.pdf"
