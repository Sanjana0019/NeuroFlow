from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.runs import router as runs_router


class MockRunsDBConn:
    def __init__(self):
        self.evaluations = {}

    async def fetchrow(self, query: str, *args):
        if "SELECT id, run_id" in query:
            run_id = args[0]
            if run_id in self.evaluations:
                return self.evaluations[run_id]
            return None
        return None

    async def fetchval(self, query: str, *args):
        if "SELECT EXISTS" in query:
            return True
        if "INSERT INTO evaluations" in query:
            return uuid4()
        return None

    async def execute(self, query: str, *args):
        if "UPDATE evaluations" in query:
            rating, eval_id = args[0], args[1]
            for r_id, eval_data in self.evaluations.items():
                if eval_data["id"] == eval_id:
                    eval_data["user_rating"] = rating


class MockRunsDBPool:
    def __init__(self, conn):
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


@pytest.fixture
def runs_client():
    app = FastAPI()
    app.include_router(runs_router)
    db_conn = MockRunsDBConn()
    app.state.db_pool = MockRunsDBPool(db_conn)
    return TestClient(app), db_conn


def test_human_rating_valid_update_with_calibration_needed(runs_client):
    """Detects calibration_needed=True when abs(overall_score - human_score) > 0.3."""
    client, db_conn = runs_client
    run_id = uuid4()
    eval_id = uuid4()

    # Automated score is 0.95 (High), Human rating is 1 (0.2 -> Low) -> diff 0.75 > 0.3
    db_conn.evaluations[run_id] = {
        "id": eval_id,
        "run_id": run_id,
        "faithfulness": 0.95,
        "answer_relevance": 0.95,
        "context_precision": 0.95,
        "context_recall": 0.95,
        "overall_score": 0.95,
        "judge_model": "gpt-4o-mini",
        "user_rating": None,
    }

    res = client.patch(f"/runs/{run_id}/rating", json={"rating": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["user_rating"] == 1
    assert data["calibration_needed"] is True


def test_human_rating_agreement_no_calibration(runs_client):
    """Detects calibration_needed=False when abs(overall_score - human_score) <= 0.3."""
    client, db_conn = runs_client
    run_id = uuid4()
    eval_id = uuid4()

    # Automated score is 0.85, Human rating is 4 (0.8) -> diff 0.05 <= 0.3
    db_conn.evaluations[run_id] = {
        "id": eval_id,
        "run_id": run_id,
        "faithfulness": 0.85,
        "answer_relevance": 0.85,
        "context_precision": 0.85,
        "context_recall": 0.85,
        "overall_score": 0.85,
        "judge_model": "gpt-4o-mini",
        "user_rating": None,
    }

    res = client.patch(f"/runs/{run_id}/rating", json={"rating": 4})
    assert res.status_code == 200
    data = res.json()
    assert data["user_rating"] == 4
    assert data["calibration_needed"] is False


def test_human_rating_invalid_out_of_bounds(runs_client):
    """Rejects rating < 1 or > 5 with 422/400 validation error."""
    client, _ = runs_client
    run_id = uuid4()

    res_zero = client.patch(f"/runs/{run_id}/rating", json={"rating": 0})
    assert res_zero.status_code in (400, 422)

    res_six = client.patch(f"/runs/{run_id}/rating", json={"rating": 6})
    assert res_six.status_code in (400, 422)
