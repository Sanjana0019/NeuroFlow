import json
import pytest

from evaluation.metrics.faithfulness import evaluate_faithfulness


class MockJudgeClient:
    def __init__(self, claims: list[str], verdicts: list[str]):
        self.claims = claims
        self.verdicts = list(verdicts)
        self.recorded_criteria = []

    async def chat(self, messages, routing_criteria=None, **kwargs):
        self.recorded_criteria.append(routing_criteria)
        system_content = messages[0].content
        if "Break down the following text" in system_content:
            return type("Res", (), {"content": json.dumps(self.claims), "model": "gpt-4o-mini"})()
        if "factual verification judge" in system_content:
            verdict = self.verdicts.pop(0) if self.verdicts else "yes"
            return type("Res", (), {"content": verdict, "model": "gpt-4o-mini"})()
        return type("Res", (), {"content": "yes", "model": "gpt-4o-mini"})()


@pytest.mark.asyncio
async def test_faithfulness_fully_supported():
    """Returns 1.0 when all extracted claims are verified with 'yes'."""
    client = MockJudgeClient(
        claims=["Paris is the capital of France.", "Paris is located in Europe."],
        verdicts=["yes", "yes"],
    )
    score = await evaluate_faithfulness(
        query="What is Paris?",
        answer="Paris is the capital of France. It is located in Europe.",
        context="Paris is the capital of France in Europe.",
        client=client,
    )
    assert score == 1.0
    assert client.recorded_criteria[0].task_type == "evaluation"


@pytest.mark.asyncio
async def test_faithfulness_partially_supported():
    """Returns 0.5 when 1 claim is yes and 1 claim is no."""
    client = MockJudgeClient(
        claims=["Earth has one moon.", "Earth has two suns."],
        verdicts=["yes", "no"],
    )
    score = await evaluate_faithfulness(
        query="Describe Earth",
        answer="Earth has one moon. Earth has two suns.",
        context="Earth has a single natural satellite called the Moon.",
        client=client,
    )
    assert score == 0.5


@pytest.mark.asyncio
async def test_faithfulness_empty_context_returns_zero():
    """Returns 0.0 when context is empty but answer contains claims."""
    score = await evaluate_faithfulness(
        query="Where is Berlin?",
        answer="Berlin is in Germany.",
        context="",
    )
    assert score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_empty_answer_returns_zero():
    """Returns 0.0 when answer is empty."""
    score = await evaluate_faithfulness(
        query="Where is Berlin?",
        answer="",
        context="Berlin is in Germany.",
    )
    assert score == 0.0
