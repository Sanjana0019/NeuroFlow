import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.api.runs import router as runs_router
from evaluation.judge import EvaluationJudge, EvaluationScore


class MockTask11DBConn:
    def __init__(self):
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.evaluations: list[dict] = []
        self.pipeline_runs: list[dict] = []
        self.pipelines: list[dict] = []

    def transaction(self):
        class Tx:
            async def __aenter__(self):
                pass
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return Tx()

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())

        if "FROM documents" in normalized:
            return self.documents

        if "FROM chunks WHERE document_id =" in normalized and "<=>" in normalized:
            doc_id = args[1]
            limit = args[2]
            return [
                {
                    "id": c["id"],
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "similarity_score": 0.92,
                }
                for c in self.chunks if str(c.get("document_id")) == str(doc_id)
            ][:limit]

        if "FROM chunks WHERE document_id =" in normalized and "ILIKE" in normalized:
            doc_id = args[0]
            limit = args[2] if len(args) > 2 else 5
            return [
                {
                    "id": c["id"],
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "similarity_score": 0.85,
                }
                for c in self.chunks if str(c.get("document_id")) == str(doc_id)
            ][:limit]

        if "FROM chunks WHERE id = ANY" in normalized:
            chunk_ids = [str(uid) for uid in args[0]]
            return [c for c in self.chunks if str(c.get("id")) in chunk_ids]

        if "FROM chunks WHERE document_id =" in normalized:
            doc_id = args[0]
            return [c for c in self.chunks if str(c.get("document_id")) == str(doc_id)]

        if "FROM evaluations e" in normalized:
            return self.evaluations

        return []

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split())

        if "FROM documents WHERE id =" in normalized:
            doc_id = str(args[0])
            for d in self.documents:
                if str(d.get("id")) == doc_id:
                    return d
            return None

        if "FROM evaluations e" in normalized and "WHERE e.run_id =" in normalized:
            run_id = str(args[0])
            for ev in self.evaluations:
                if str(ev.get("run_id")) == run_id:
                    return ev
            return None

        return None

    async def execute(self, query: str, *args):
        pass


class MockTask11DBPool:
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
def task11_client_and_db():
    app = FastAPI()
    app.include_router(ingest_router)
    app.include_router(runs_router)
    app.include_router(query_router)

    db_conn = MockTask11DBConn()
    db_pool = MockTask11DBPool(db_conn)

    app.state.db_pool = db_pool
    app.state.redis = None
    app.state.arq_redis = None
    app.state.neuroflow_client = None

    client = TestClient(app)
    return client, db_conn


def test_list_documents_and_chunks(task11_client_and_db):
    """GET /documents returns document list and GET /documents/{id}/chunks returns chunks."""
    client, db_conn = task11_client_and_db

    doc_id = uuid4()
    db_conn.documents.append({
        "id": doc_id,
        "filename": "annual_report.pdf",
        "source_type": "pdf",
        "status": "completed",
        "chunk_count": 2,
        "metadata": json.dumps({"pages": 12}),
        "created_at": datetime.now(timezone.utc),
    })

    db_conn.chunks.extend([
        {
            "id": uuid4(),
            "document_id": doc_id,
            "chunk_index": 0,
            "content": "Section 1: Revenue grew by 25%.",
            "token_count": 8,
            "metadata": {},
        },
        {
            "id": uuid4(),
            "document_id": doc_id,
            "chunk_index": 1,
            "content": "Section 2: Operating margin expanded to 18%.",
            "token_count": 8,
            "metadata": {},
        },
    ])

    # 1. List Documents
    doc_res = client.get("/documents")
    assert doc_res.status_code == 200
    docs = doc_res.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "annual_report.pdf"
    assert docs[0]["chunk_count"] == 2

    # 2. Get Document Chunks
    chunk_res = client.get(f"/documents/{doc_id}/chunks")
    assert chunk_res.status_code == 200
    chunks = chunk_res.json()
    assert len(chunks) == 2
    assert "Revenue grew" in chunks[0]["content"]


def test_find_similar_chunks(task11_client_and_db):
    """POST /documents/{id}/similar finds top similar chunks in document."""
    client, db_conn = task11_client_and_db

    doc_id = uuid4()
    db_conn.chunks.append({
        "id": uuid4(),
        "document_id": doc_id,
        "chunk_index": 0,
        "content": "Our Q3 cash flow exceeded forecasts by $10M.",
        "token_count": 10,
        "metadata": {},
    })

    res = client.post(
        f"/documents/{doc_id}/similar",
        json={"query": "cash flow forecasts", "limit": 3},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["similarity_score"] > 0.5


def test_list_evaluations_and_single_run_eval(task11_client_and_db):
    """GET /evaluations lists evaluations and GET /runs/{run_id}/evaluation returns single run score."""
    client, db_conn = task11_client_and_db

    run_id = uuid4()
    chunk_id = uuid4()
    db_conn.chunks.append({
        "id": chunk_id,
        "content": "Relevant background fact.",
        "chunk_index": 0,
    })

    db_conn.evaluations.append({
        "id": uuid4(),
        "run_id": run_id,
        "query": "What is the policy limit?",
        "generation": "Based on [Source 1], the policy limit is $1M.",
        "pipeline_name": "Production RAG",
        "faithfulness": 0.95,
        "answer_relevance": 0.92,
        "context_precision": 0.88,
        "context_recall": 0.85,
        "overall_score": 0.91,
        "judge_model": "gpt-4o-mini",
        "user_rating": 5,
        "evaluated_at": datetime.now(timezone.utc),
        "retrieved_chunk_ids": [chunk_id],
    })

    # 1. List evaluations
    list_res = client.get("/evaluations?min_overall=0.8")
    assert list_res.status_code == 200
    evals = list_res.json()
    assert len(evals) == 1
    assert evals[0]["overall_score"] == 0.91
    assert len(evals[0]["chunks"]) == 1

    # 2. Get single run evaluation
    single_res = client.get(f"/runs/{run_id}/evaluation")
    assert single_res.status_code == 200
    assert single_res.json()["run_id"] == str(run_id)
    assert single_res.json()["faithfulness"] == 0.95


@pytest.mark.asyncio
async def test_judge_publishes_to_redis_evaluations_channel():
    """Test that EvaluationJudge publishes evaluation events to Redis evaluations:new channel."""
    mock_redis = AsyncMock()
    mock_client = MagicMock()
    mock_client.redis = mock_redis
    mock_db_pool = MagicMock()

    class MockConn:
        def transaction(self):
            class Tx:
                async def __aenter__(self):
                    pass
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            return Tx()

        async def execute(self, query, *args):
            pass

    class MockCtx:
        async def __aenter__(self):
            return MockConn()
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_db_pool.acquire.return_value = MockCtx()

    judge = EvaluationJudge(client=mock_client, db_pool=mock_db_pool)
    score = EvaluationScore(
        faithfulness=0.9,
        answer_relevance=0.85,
        context_precision=0.8,
        context_recall=0.75,
        overall_score=0.84,
        judge_model="gpt-4o-mini",
        is_training_candidate=True,
    )

    await judge._persist_evaluation(
        run_uuid=uuid4(),
        score=score,
        query="Test query",
        answer="Test answer",
    )

    mock_redis.publish.assert_called_once()
    args, _ = mock_redis.publish.call_args
    assert args[0] == "evaluations:new"
    payload = json.loads(args[1])
    assert payload["overall_score"] == 0.84
