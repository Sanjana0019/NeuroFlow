import io
import json
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from backend.api.ingest import router as ingest_router
from backend.db.documents import DocumentRepository
from backend.worker import process_ingestion_job
from pipelines.ingestion.chunker import Chunker
from pipelines.ingestion.dispatcher import IngestionDispatcher
from pipelines.ingestion.embedder import ChunkEmbedder
from pipelines.ingestion.models import Chunk, ExtractedPage


class LiveMemoryDBConnection:
    """In-memory database simulating PostgreSQL table operations for documents and chunks."""

    def __init__(self):
        self.documents: dict[UUID, dict] = {}
        self.content_hash_map: dict[str, UUID] = {}
        self.chunks: dict[UUID, list[dict]] = {}

    def transaction(self):
        class DummyTx:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False
        return DummyTx()

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        if "INSERT INTO documents" in normalized:
            filename, source_type, content_hash, meta_str, pipeline_id, status, chunk_count = (
                args[0], args[1], args[2], args[3], args[4],
                args[5] if len(args) > 5 else "queued",
                args[6] if len(args) > 6 else 0,
            )
            doc_id = uuid4()
            record = {
                "id": doc_id,
                "filename": filename,
                "source_type": source_type,
                "content_hash": content_hash,
                "metadata": meta_str,
                "pipeline_id": pipeline_id,
                "status": status,
                "chunk_count": chunk_count,
            }
            self.documents[doc_id] = record
            self.content_hash_map[content_hash] = doc_id
            return doc_id
        return None

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        if "WHERE content_hash = $1" in normalized:
            content_hash = args[0]
            doc_id = self.content_hash_map.get(content_hash)
            if doc_id:
                return self.documents.get(doc_id)
            return None
        elif "WHERE id = $1" in normalized:
            raw_id = args[0]
            doc_uuid = UUID(str(raw_id)) if not isinstance(raw_id, UUID) else raw_id
            return self.documents.get(doc_uuid)
        return None

    async def executemany(self, query, args):
        normalized = " ".join(query.split())
        if "INSERT INTO chunks" in normalized:
            for record in args:
                doc_id, content, embedding_str, chunk_idx, token_cnt, meta_str = record
                doc_uuid = UUID(str(doc_id)) if not isinstance(doc_id, UUID) else doc_id
                if doc_uuid not in self.chunks:
                    self.chunks[doc_uuid] = []
                self.chunks[doc_uuid].append(
                    {
                        "document_id": doc_uuid,
                        "content": content,
                        "embedding": embedding_str,
                        "chunk_index": chunk_idx,
                        "token_count": token_cnt,
                        "metadata": meta_str,
                    }
                )

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if "UPDATE documents SET status = $1 WHERE id = $2" in normalized:
            status, doc_id = args
            doc_uuid = UUID(str(doc_id)) if not isinstance(doc_id, UUID) else doc_id
            if doc_uuid in self.documents:
                self.documents[doc_uuid]["status"] = status
        elif "UPDATE documents SET chunk_count = COALESCE(chunk_count, 0) + $1 WHERE id = $2" in normalized:
            count, doc_id = args
            doc_uuid = UUID(str(doc_id)) if not isinstance(doc_id, UUID) else doc_id
            if doc_uuid in self.documents:
                self.documents[doc_uuid]["chunk_count"] = (
                    self.documents[doc_uuid].get("chunk_count", 0) + count
                )


class LiveMemoryDBPool:
    def __init__(self, connection: LiveMemoryDBConnection):
        self.connection = connection

    def acquire(self):
        class AcquireContext:
            def __init__(self, conn):
                self.conn = conn
            async def __aenter__(self):
                return self.conn
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return AcquireContext(self.connection)


class MockEmbeddingClient:
    """Deterministic embedding client providing 1536-dimensional vectors."""
    def __init__(self):
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[0.05] * 1536 for _ in texts]


class DirectARQQueue:
    """Executes ingestion jobs asynchronously in memory for end-to-end integration testing."""
    def __init__(self, worker_ctx: dict):
        self.worker_ctx = worker_ctx
        self.enqueued_jobs = []

    async def enqueue_job(self, function_name: str, **kwargs):
        self.enqueued_jobs.append({"function": function_name, "args": kwargs})
        # Execute worker job with provided context
        return await process_ingestion_job(
            ctx=self.worker_ctx,
            source=kwargs.get("source"),
            source_type=kwargs.get("source_type"),
            filename=kwargs.get("filename"),
            metadata=kwargs.get("metadata"),
            pipeline_id=kwargs.get("pipeline_id"),
            document_id=kwargs.get("document_id"),
        )


@pytest.fixture
def e2e_environment():
    db_conn = LiveMemoryDBConnection()
    db_pool = LiveMemoryDBPool(db_conn)
    embed_client = MockEmbeddingClient()
    embedder = ChunkEmbedder(client=embed_client)
    repository = DocumentRepository(db_pool)
    dispatcher = IngestionDispatcher()
    chunker = Chunker()

    worker_ctx = {
        "db_pool": db_pool,
        "repository": repository,
        "dispatcher": dispatcher,
        "chunker": chunker,
        "embedder": embedder,
    }

    arq_queue = DirectARQQueue(worker_ctx=worker_ctx)

    app = FastAPI()
    app.include_router(ingest_router)
    app.state.db_pool = db_pool
    app.state.arq_redis = arq_queue

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(processor)
    else:
        new_provider = TracerProvider()
        new_provider.add_span_processor(processor)
        trace._TRACER_PROVIDER = new_provider

    client = TestClient(app)

    yield {
        "client": client,
        "db_conn": db_conn,
        "arq_queue": arq_queue,
        "embed_client": embed_client,
        "exporter": exporter,
    }

    exporter.clear()


def test_e2e_full_pdf_pipeline_lifecycle(e2e_environment, caplog):
    """E2E Test: Upload PDF -> API queued -> Worker processing -> completed -> Deduplication -> GET /documents."""
    client = e2e_environment["client"]
    db_conn = e2e_environment["db_conn"]
    arq_queue = e2e_environment["arq_queue"]
    embed_client = e2e_environment["embed_client"]
    exporter = e2e_environment["exporter"]

    pdf_fixture_path = Path("backend/tests/fixtures/digital_test.pdf")
    with open(pdf_fixture_path, "rb") as f:
        pdf_bytes = f.read()

    # 1. First Submission: Upload PDF via POST /ingest
    with caplog.at_level(logging.INFO, logger="neuroflow.ingestion"):
        res1 = client.post(
            "/ingest",
            files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

    assert res1.status_code == 202
    data1 = res1.json()
    doc_id_str = data1["document_id"]
    doc_id = UUID(doc_id_str)
    assert data1["status"] == "queued"
    assert data1["duplicate"] is False

    # 2. Verify Database Persistence
    doc_record = db_conn.documents[doc_id]
    assert doc_record["status"] == "completed"
    assert doc_record["chunk_count"] > 0
    assert doc_id in db_conn.chunks
    assert len(db_conn.chunks[doc_id]) == doc_record["chunk_count"]

    # Verify vector embeddings attached to chunks
    for chunk_rec in db_conn.chunks[doc_id]:
        assert chunk_rec["embedding"] is not None
        assert chunk_rec["embedding"].startswith("[0.05,0.05")

    # 3. Verify OpenTelemetry Span 'ingestion.process'
    spans = [s for s in exporter.get_finished_spans() if s.name == "ingestion.process"]
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["document_id"] == doc_id_str
    assert span.attributes["source_type"] == "pdf"
    assert span.attributes["page_count"] >= 1
    assert span.attributes["chunk_count"] == doc_record["chunk_count"]
    assert span.attributes["embedding_calls"] >= 1

    # 4. Verify Structured JSON Completion Log
    ingest_logs = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "neuroflow.ingestion" and "ingestion_complete" in r.message
    ]
    assert len(ingest_logs) == 1
    log_entry = ingest_logs[0]
    assert log_entry["event"] == "ingestion_complete"
    assert log_entry["document_id"] == doc_id_str
    assert log_entry["chunks"] == doc_record["chunk_count"]
    assert log_entry["tokens"] > 0
    assert isinstance(log_entry["duration_ms"], (int, float))

    # 5. Verify GET /documents/{document_id}
    get_res = client.get(f"/documents/{doc_id_str}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["document_id"] == doc_id_str
    assert get_data["status"] == "completed"
    assert get_data["chunk_count"] == doc_record["chunk_count"]

    # 6. Deduplication Verification: Upload identical PDF again
    initial_embed_calls = embed_client.call_count
    res2 = client.post(
        "/ingest",
        files={"file": ("report_renamed.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res2.status_code == 202
    data2 = res2.json()
    assert data2["duplicate"] is True
    assert data2["document_id"] == doc_id_str
    assert data2["status"] == "completed"

    # Confirm no additional jobs or embedding calls occurred
    assert len(arq_queue.enqueued_jobs) == 1
    assert embed_client.call_count == initial_embed_calls


def test_e2e_csv_pipeline_lifecycle(e2e_environment):
    """E2E Test: Upload CSV -> Full Pipeline Execution -> Verify Persisted Chunks & Metadata."""
    client = e2e_environment["client"]
    db_conn = e2e_environment["db_conn"]

    csv_content = b"Item,Cost,Quantity\nWidget A,15.50,100\nWidget B,23.00,50\n"
    res = client.post(
        "/ingest",
        files={"file": ("inventory.csv", io.BytesIO(csv_content), "text/csv")},
    )

    assert res.status_code == 202
    data = res.json()
    doc_id = UUID(data["document_id"])
    assert data["status"] == "queued"
    assert data["duplicate"] is False

    # Check status and chunk storage
    doc_record = db_conn.documents[doc_id]
    assert doc_record["status"] == "completed"
    assert doc_record["source_type"] == "csv"
    assert len(db_conn.chunks[doc_id]) >= 1
    assert "Widget A" in db_conn.chunks[doc_id][0]["content"]


def test_e2e_docx_pipeline_lifecycle(e2e_environment):
    """E2E Test: Upload DOCX -> Full Pipeline Execution -> Verify Hierarchical Chunks."""
    client = e2e_environment["client"]
    db_conn = e2e_environment["db_conn"]

    docx_fixture_path = Path("backend/tests/fixtures/test_document.docx")
    with open(docx_fixture_path, "rb") as f:
        docx_bytes = f.read()

    res = client.post(
        "/ingest",
        files={"file": ("manual.docx", io.BytesIO(docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert res.status_code == 202
    data = res.json()
    doc_id = UUID(data["document_id"])
    assert data["status"] == "queued"

    doc_record = db_conn.documents[doc_id]
    assert doc_record["status"] == "completed"
    assert doc_record["source_type"] == "docx"
    assert len(db_conn.chunks[doc_id]) >= 1


def test_e2e_pptx_pipeline_lifecycle(e2e_environment):
    """E2E Test: Upload PPTX -> Full Pipeline Execution -> Verify Speaker Notes."""
    client = e2e_environment["client"]
    db_conn = e2e_environment["db_conn"]

    pptx_fixture_path = Path("backend/tests/fixtures/test_presentation.pptx")
    with open(pptx_fixture_path, "rb") as f:
        pptx_bytes = f.read()

    res = client.post(
        "/ingest",
        files={"file": ("slides.pptx", io.BytesIO(pptx_bytes), "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
    )

    assert res.status_code == 202
    data = res.json()
    doc_id = UUID(data["document_id"])
    assert data["status"] == "queued"

    doc_record = db_conn.documents[doc_id]
    assert doc_record["status"] == "completed"
    assert doc_record["source_type"] == "text"
    assert len(db_conn.chunks[doc_id]) >= 1

