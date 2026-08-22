from uuid import uuid4
import pytest

from backend.providers.base import GenerationResult
from pipelines.retrieval.models import RetrievalResult
from pipelines.retrieval.reranker import Reranker


class MockRerankClient:
    def __init__(self, score_map: dict[str, str]):
        self.score_map = score_map
        self.call_count = 0

    async def chat(self, messages, routing_criteria, **kwargs):
        self.call_count += 1
        user_prompt = messages[-1].content
        for key, score_str in self.score_map.items():
            if key in user_prompt:
                return GenerationResult(
                    content=score_str,
                    model="gpt-4o-mini",
                    input_tokens=40,
                    output_tokens=5,
                    latency_ms=80.0,
                    cost_usd=0.00005,
                    finish_reason="stop",
                )
        return GenerationResult(
            content="5.0",
            model="gpt-4o-mini",
            input_tokens=40,
            output_tokens=5,
            latency_ms=80.0,
            cost_usd=0.00005,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_reranker_scores_and_sorts_candidates():
    """Reranker queries LLM in parallel and sorts candidates by relevance score."""
    c1 = RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="Low relevance text about apples", score=0.9, filename="a.pdf")
    c2 = RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="High relevance: Attention is all you need in transformers", score=0.5, filename="b.pdf")

    score_map = {
        "apples": "1.5",
        "Attention is all you need": "9.5",
    }
    client = MockRerankClient(score_map)
    reranker = Reranker(client=client, concurrency=5)

    results = await reranker.rerank(
        query="Explain transformer attention mechanism",
        candidates=[c1, c2],
    )

    assert len(results) == 2
    # c2 was scored 9.5 and should now be ranked #1
    assert results[0].chunk_id == c2.chunk_id
    assert results[0].score == 9.5
    assert results[0].rank == 1
    assert results[0].source == "reranked"

    # c1 was scored 1.5 and is ranked #2
    assert results[1].chunk_id == c1.chunk_id
    assert results[1].score == 1.5
    assert results[1].rank == 2


@pytest.mark.asyncio
async def test_reranker_handles_malformed_output():
    """Reranker parses messy text or falls back safely when score is not standard."""
    c1 = RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="Passage text", score=0.6, filename="doc.pdf")
    score_map = {"Passage text": "The score is 8.7 out of 10"}
    client = MockRerankClient(score_map)
    reranker = Reranker(client=client)

    results = await reranker.rerank("Query", [c1])
    assert len(results) == 1
    assert results[0].score == 8.7


@pytest.mark.asyncio
async def test_reranker_fallback_without_client():
    """Reranker falls back to word overlap heuristic when no client is provided."""
    c1 = RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="Attention mechanism in transformers", score=0.5)
    c2 = RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="Baking sourdough bread at home", score=0.8)

    reranker = Reranker(client=None)
    results = await reranker.rerank("transformers attention", [c1, c2])

    assert results[0].chunk_id == c1.chunk_id
    assert results[0].score > results[1].score
