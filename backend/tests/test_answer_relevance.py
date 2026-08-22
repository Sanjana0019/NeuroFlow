import json
import pytest

from evaluation.metrics.answer_relevance import evaluate_answer_relevance


class MockRelevanceClient:
    def __init__(self, questions: list[str]):
        self.questions = questions

    async def chat(self, messages, routing_criteria=None, **kwargs):
        return type("Res", (), {"content": json.dumps(self.questions), "model": "gpt-4o-mini"})()

    async def embed(self, texts: list[str]):
        # Return identical embeddings for identical text and scaled for others
        base_vec = [1.0, 0.0, 0.0]
        return [base_vec for _ in texts]


@pytest.mark.asyncio
async def test_answer_relevance_high_similarity():
    """Returns 1.0 when generated questions have identical embeddings to query."""
    client = MockRelevanceClient(
        questions=[
            "What is the speed of light?",
            "How fast does light travel in vacuum?",
        ]
    )
    score = await evaluate_answer_relevance(
        query="What is the speed of light?",
        answer="Light travels at 300,000 km/s in vacuum.",
        client=client,
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_answer_relevance_empty_query_returns_zero():
    """Returns 0.0 when query is empty."""
    score = await evaluate_answer_relevance(
        query="",
        answer="Some answer",
    )
    assert score == 0.0
