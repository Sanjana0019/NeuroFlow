import asyncio
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4
import pytest

import evaluation.judge
from evaluation.judge import EvaluationJudge


class MockFullJudgeClient:
    async def chat(self, messages, routing_criteria=None, **kwargs):
        system_content = messages[0].content
        if "Break down the following text" in system_content:
            return type("Res", (), {"content": '["Fact A"]', "model": "gpt-4o-mini"})()
        if "factual verification judge" in system_content:
            return type("Res", (), {"content": "yes", "model": "gpt-4o-mini"})()
        if "generate 3 to 5 distinct questions" in system_content:
            return type("Res", (), {"content": '["What is Fact A?"]', "model": "gpt-4o-mini"})()
        if "useful and relevant" in system_content:
            return type("Res", (), {"content": "yes", "model": "gpt-4o-mini"})()
        if "attributed to the provided reference" in system_content:
            return type("Res", (), {"content": "yes", "model": "gpt-4o-mini"})()
        return type("Res", (), {"content": "yes", "model": "gpt-4o-mini"})()

    async def embed(self, texts: list[str]):
        return [[1.0, 0.0, 0.0] for _ in texts]


class MockEvalDBConn:
    def __init__(self):
        self.evaluations = []
        self.training_pairs = []

    def transaction(self):
        class TxCtx:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return TxCtx()

    async def execute(self, query: str, *args):
        if "INSERT INTO evaluations" in query:
            self.evaluations.append({
                "run_id": args[0],
                "faithfulness": args[1],
                "answer_relevance": args[2],
                "context_precision": args[3],
                "context_recall": args[4],
                "overall_score": args[5],
                "judge_model": args[6],
            })
        elif "INSERT INTO training_pairs" in query:
            self.training_pairs.append({
                "run_id": args[0],
                "system_prompt": args[1],
                "user_message": args[2],
                "assistant_message": args[3],
                "quality_score": args[4],
            })


class MockEvalDBPool:
    def __init__(self, conn: MockEvalDBConn):
        self.conn = conn

    def acquire(self):
        class Ctx:
            def __init__(self, c):
                self.c = c
            async def __aenter__(self):
                return self.c
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return Ctx(self.conn)


@pytest.mark.asyncio
async def test_evaluation_judge_concurrency_and_high_score_training_pair():
    """Judge runs metrics concurrently, records OpenTelemetry span, persists eval, and creates training pair."""
    client = MockFullJudgeClient()
    db_conn = MockEvalDBConn()
    db_pool = MockEvalDBPool(db_conn)

    mock_span = MagicMock()
    mock_span_context = MagicMock()
    mock_span_context.__enter__.return_value = mock_span
    mock_span_context.__exit__.return_value = False

    judge = EvaluationJudge(client=client, db_pool=db_pool)
    run_id = uuid4()

    with patch.object(evaluation.judge.tracer, "start_as_current_span", return_value=mock_span_context) as mock_tracer:
        score = await judge.evaluate_run(
            run_id=run_id,
            query="What is Fact A?",
            answer="Fact A is true.",
            context=["Fact A is true."],
        )

        # 1. Verification of Scores
        assert score.faithfulness == 1.0
        assert score.answer_relevance == 1.0
        assert score.context_precision == 1.0
        assert score.context_recall == 1.0
        assert score.overall_score == 1.0
        assert score.is_training_candidate is True

        # 2. Database Persistence
        assert len(db_conn.evaluations) == 1
        assert db_conn.evaluations[0]["run_id"] == run_id
        assert db_conn.evaluations[0]["overall_score"] == 1.0

        # 3. Training Pair Creation (since overall_score 1.0 > 0.8)
        assert len(db_conn.training_pairs) == 1
        assert db_conn.training_pairs[0]["run_id"] == run_id
        assert db_conn.training_pairs[0]["user_message"] == "What is Fact A?"
        assert db_conn.training_pairs[0]["quality_score"] == 1.0

        # 4. OpenTelemetry Span Verification
        mock_tracer.assert_called_once_with("evaluation.judge")
        mock_span.set_attribute.assert_any_call("run_id", str(run_id))
        mock_span.set_attribute.assert_any_call("overall_score", 1.0)
        mock_span.set_attribute.assert_any_call("faithfulness", 1.0)


@pytest.mark.asyncio
async def test_evaluation_judge_low_score_no_training_pair():
    """Does not create training pair when overall_score <= 0.8."""
    # When context is empty -> faithfulness=0, recall=0 -> overall <= 0.8
    client = MockFullJudgeClient()
    db_conn = MockEvalDBConn()
    db_pool = MockEvalDBPool(db_conn)

    judge = EvaluationJudge(client=client, db_pool=db_pool)
    run_id = uuid4()

    score = await judge.evaluate_run(
        run_id=run_id,
        query="What is Fact A?",
        answer="Fact A is true.",
        context=[],
    )

    assert score.overall_score <= 0.8
    assert score.is_training_candidate is False
    assert len(db_conn.evaluations) == 1
    assert len(db_conn.training_pairs) == 0
