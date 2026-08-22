import asyncio
from uuid import UUID, uuid4
import pytest

from pipelines.generation.generator import Generator
from pipelines.retrieval.models import AssembledContext, RetrievalResult


class MockStreamProvider:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.model = "mock-gpt-4o-mini"
        self.stream_calls = []

    async def stream(self, messages, **kwargs):
        self.stream_calls.append(messages)
        for token in self.tokens:
            yield token


class MockClientWithStream:
    def __init__(self, provider: MockStreamProvider):
        self.provider = provider
        self.providers = {"mock": provider}

    async def stream(self, messages, criteria, **kwargs):
        return self.provider.stream(messages), self.provider.model


class MockDBConn:
    def __init__(self):
        self.pipeline_runs: dict[UUID, dict] = {}
        self.pipeline_id = uuid4()

    async def fetchrow(self, query: str, *args):
        if "SELECT id FROM pipelines" in query:
            return {"id": self.pipeline_id}
        return None

    async def fetchval(self, query: str, *args):
        if "INSERT INTO pipeline_runs" in query:
            run_id = uuid4()
            p_id, q_text, chunk_ids = args[0], args[1], args[2]
            self.pipeline_runs[run_id] = {
                "id": run_id,
                "pipeline_id": p_id,
                "query": q_text,
                "retrieved_chunk_ids": chunk_ids,
                "status": "running",
            }
            return run_id
        return None

    async def execute(self, query: str, *args):
        if "UPDATE pipeline_runs" in query:
            gen_text, in_tok, out_tok, lat, mod, stat, r_id = (
                args[0], args[1], args[2], args[3], args[4], args[5], args[6]
            )
            if r_id in self.pipeline_runs:
                self.pipeline_runs[r_id].update(
                    {
                        "generation": gen_text,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "latency_ms": lat,
                        "model_used": mod,
                        "status": stat,
                    }
                )


class MockDBPool:
    def __init__(self, conn: MockDBConn):
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


class MockARQRedis:
    def __init__(self):
        self.enqueued_jobs = []

    async def enqueue_job(self, func_name: str, **kwargs):
        self.enqueued_jobs.append({"function": func_name, "args": kwargs})


@pytest.mark.asyncio
async def test_generator_streaming_and_persistence():
    """Generator streams tokens, updates DB with complete status and enqueues evaluation job."""
    tokens = ["Based ", "on ", "[Source 1], ", "attention ", "is ", "effective."]
    provider = MockStreamProvider(tokens)
    client = MockClientWithStream(provider)
    db_conn = MockDBConn()
    db_pool = MockDBPool(db_conn)
    arq_redis = MockARQRedis()

    generator = Generator(client=client)

    c1 = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Attention mechanisms model dependencies.",
        filename="paper.pdf",
        page_number=1,
    )
    assembled = AssembledContext(
        context="[Source 1 — paper.pdf, page 1]\nAttention mechanisms model dependencies.",
        chunks_used=[c1],
        total_tokens=15,
    )

    events = []
    async for event in generator.stream_generation(
        query="How does attention work?",
        assembled_context=assembled,
        db_pool=db_pool,
        arq_redis=arq_redis,
    ):
        events.append(event)

    # 1. Verify token events
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 6
    deltas = [e["delta"] for e in token_events]
    assert "".join(deltas) == "Based on [Source 1], attention is effective."

    # 2. Verify done event
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    run_id = UUID(done["run_id"])
    assert done["generation"] == "Based on [Source 1], attention is effective."
    assert len(done["citations"]) == 1
    assert done["citations"][0]["reference"] == "[Source 1]"
    assert done["citations"][0]["chunk_id"] == str(c1.chunk_id)

    # 3. Verify Database Persistence (pipeline_runs was updated to complete)
    assert run_id in db_conn.pipeline_runs
    persisted = db_conn.pipeline_runs[run_id]
    assert persisted["status"] == "complete"
    assert persisted["generation"] == "Based on [Source 1], attention is effective."
    assert persisted["input_tokens"] > 0
    assert persisted["output_tokens"] > 0
    assert persisted["latency_ms"] >= 0

    # 4. Verify asynchronous evaluation job enqueueing (wait 50ms for background task)
    await asyncio.sleep(0.05)
    assert len(arq_redis.enqueued_jobs) == 1
    assert arq_redis.enqueued_jobs[0]["function"] == "evaluate_pipeline_run"
    assert arq_redis.enqueued_jobs[0]["args"]["run_id"] == str(run_id)
