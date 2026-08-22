import json
from uuid import UUID, uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.query import router as query_router


class MockAPIProvider:
    def __init__(self):
        self.model = "gpt-4o-mini"

    async def complete(self, messages, **kwargs):
        system_content = messages[0].content if messages else ""
        if "query understanding" in system_content:
            content = json.dumps({
                "expanded_queries": ["transformer self-attention mechanism"],
                "metadata_filters": {},
                "query_type": "factual",
            })
        else:
            content = "8.5"

        return type(
            "Res",
            (),
            {
                "content": content,
                "model": "gpt-4o-mini",
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.00001,
                "latency_ms": 10.0,
            },
        )()

    async def stream(self, messages, **kwargs):
        yield "According "
        yield "to "
        yield "[Source 1], "
        yield "the "
        yield "model "
        yield "works."

    async def embed(self, texts):
        return [[0.05] * 1536 for _ in texts]


class MockAPIClient:
    def __init__(self):
        self.provider = MockAPIProvider()
        self.providers = {"openai": self.provider, "mock": self.provider}

    async def chat(self, messages, routing_criteria, **kwargs):
        return await self.provider.complete(messages)

    async def stream(self, messages, routing_criteria, **kwargs):
        return self.provider.stream(messages), "gpt-4o-mini"

    async def embed(self, texts):
        return await self.provider.embed(texts)


class MockAPIDBConn:
    def __init__(self):
        self.pipeline_runs = {}
        self.pipeline_id = uuid4()

    async def fetch(self, query: str, *args):
        c_id = uuid4()
        d_id = uuid4()
        return [
            {
                "id": c_id,
                "document_id": d_id,
                "content": "Transformers utilize multi-head self-attention.",
                "chunk_index": 0,
                "token_count": 8,
                "metadata": json.dumps({"page_number": 2}),
                "filename": "model_paper.pdf",
                "score": 0.95,
            }
        ]

    async def fetchrow(self, query: str, *args):
        if "SELECT id FROM pipelines" in query:
            return {"id": self.pipeline_id}
        return None

    async def fetchval(self, query: str, *args):
        if "INSERT INTO pipeline_runs" in query:
            run_id = uuid4()
            self.pipeline_runs[run_id] = {"status": "running"}
            return run_id
        return None

    async def execute(self, query: str, *args):
        pass


class MockAPIDBPool:
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
def query_client():
    app = FastAPI()
    app.include_router(query_router)

    db_conn = MockAPIDBConn()
    db_pool = MockAPIDBPool(db_conn)
    client_obj = MockAPIClient()

    app.state.db_pool = db_pool
    app.state.neuroflow_client = client_obj
    app.state.arq_redis = None

    test_client = TestClient(app)
    return test_client


def test_post_query_non_streaming(query_client):
    """POST /query with stream=false returns complete JSON response with citations and metadata."""
    res = query_client.post(
        "/query",
        json={"query": "Explain self-attention", "stream": False},
    )

    assert res.status_code == 200
    data = res.json()

    assert "run_id" in data
    assert data["query"] == "Explain self-attention"
    assert "According to [Source 1]" in data["generation"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["reference"] == "[Source 1]"
    assert data["citations"][0]["document_name"] == "model_paper.pdf"
    assert data["citations"][0]["page_number"] == 2
    assert len(data["sources"]) >= 1
    assert data["status"] == "complete"


def test_post_query_streaming_and_sse_events(query_client):
    """POST /query with stream=true registers run and GET /query/{run_id}/stream streams SSE events."""
    # 1. Start streaming run
    res = query_client.post(
        "/query",
        json={"query": "Explain self-attention", "stream": True},
    )
    assert res.status_code == 200
    start_data = res.json()
    assert "run_id" in start_data
    assert start_data["status"] == "started"
    run_id = start_data["run_id"]

    # 2. Connect to SSE stream
    stream_res = query_client.get(f"/query/{run_id}/stream")
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]

    lines = stream_res.text.split("\n")
    data_lines = [l.replace("data: ", "").strip() for l in lines if l.startswith("data:")]

    events = [json.loads(l) for l in data_lines if l]
    event_types = [e["type"] for e in events]

    assert "retrieval_start" in event_types
    assert "retrieval_complete" in event_types
    assert "token" in event_types
    assert "done" in event_types

    done_event = [e for e in events if e["type"] == "done"][0]
    assert len(done_event["citations"]) == 1
    assert done_event["citations"][0]["document_name"] == "model_paper.pdf"


def test_query_missing_text_rejected(query_client):
    """Empty query returns 400 Bad Request."""
    res = query_client.post("/query", json={"query": "   "})
    assert res.status_code == 400


def test_stream_nonexistent_run_returns_404(query_client):
    """Attempting to stream a nonexistent run_id returns 404."""
    res = query_client.get(f"/query/{uuid4()}/stream")
    assert res.status_code == 404
