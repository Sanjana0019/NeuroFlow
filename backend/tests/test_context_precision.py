import pytest

from evaluation.metrics.context_precision import evaluate_context_precision


class MockPrecisionClient:
    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)

    async def chat(self, messages, routing_criteria=None, **kwargs):
        verdict = self.verdicts.pop(0) if self.verdicts else "yes"
        return type("Res", (), {"content": verdict, "model": "gpt-4o-mini"})()


@pytest.mark.asyncio
async def test_context_precision_weighting():
    """Weights earlier ranked chunks more heavily using sum(useful[i]*(1/i)) / sum(1/i)."""
    # Chunk 1 (rank 1): yes, Chunk 2 (rank 2): no
    # Score = (1.0 * 1/1 + 0.0 * 1/2) / (1/1 + 1/2) = 1.0 / 1.5 = 0.6667
    client = MockPrecisionClient(verdicts=["yes", "no"])
    score = await evaluate_context_precision(
        query="Tell me about transformers",
        chunks=["Passage 1 with relevant info", "Passage 2 with irrelevant noise"],
        answer="Transformers use self-attention.",
        client=client,
    )
    assert score == 0.6667


@pytest.mark.asyncio
async def test_context_precision_empty_chunks_returns_zero():
    """Returns 0.0 when chunks list is empty."""
    score = await evaluate_context_precision(
        query="Tell me about transformers",
        chunks=[],
        answer="Transformers use self-attention.",
    )
    assert score == 0.0
