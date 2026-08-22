import pytest

from evaluation.metrics.context_recall import evaluate_context_recall


class MockRecallClient:
    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)

    async def chat(self, messages, routing_criteria=None, **kwargs):
        verdict = self.verdicts.pop(0) if self.verdicts else "yes"
        return type("Res", (), {"content": verdict, "model": "gpt-4o-mini"})()


@pytest.mark.asyncio
async def test_context_recall_sentence_attribution():
    """Computes ratio of sentences in answer that can be attributed to context."""
    # 2 sentences: 1 yes, 1 no -> 0.5
    client = MockRecallClient(verdicts=["yes", "no"])
    score = await evaluate_context_recall(
        query="What is python?",
        chunks=["Python was created by Guido van Rossum."],
        answer="Python was created by Guido van Rossum. It was released in 1800.",
        client=client,
    )
    assert score == 0.5


@pytest.mark.asyncio
async def test_context_recall_empty_answer_returns_zero():
    """Returns 0.0 when answer is empty."""
    score = await evaluate_context_recall(
        query="What is python?",
        chunks=["Python context"],
        answer="",
    )
    assert score == 0.0
