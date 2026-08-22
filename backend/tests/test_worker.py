import json
import logging
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from backend.worker import process_ingestion_job
from pipelines.ingestion.models import Chunk, ExtractedPage


class FakeDispatcher:
    def __init__(self, pages=None, should_fail=False):
        self.pages = pages or [
            ExtractedPage(page_number=1, content="Extracted page 1", content_type="text"),
            ExtractedPage(page_number=2, content="Extracted page 2", content_type="text"),
        ]
        self.should_fail = should_fail
        self.calls = []

    def dispatch(self, source: str):
        if self.should_fail:
            raise ValueError(f"Extractor failed for {source}")
        self.calls.append(source)
        return self.pages


class FakeAsyncDispatcher:
    def __init__(self, pages=None):
        self.pages = pages or [
            ExtractedPage(page_number=1, content="Async page", content_type="text"),
        ]
        self.calls = []

    async def dispatch(self, source: str):
        self.calls.append(source)
        return self.pages


class FakeChunker:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    def chunk(self, pages, document_id=None, strategy="auto"):
        if self.should_fail:
            raise RuntimeError("Chunker failed")
        self.calls.append((pages, document_id))
        return [
            Chunk(content=p.content, chunk_index=i, token_count=10, page_number=p.page_number)
            for i, p in enumerate(pages)
        ]


class FakeEmbedder:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    async def embed_chunks(self, chunks):
        if self.should_fail:
            raise RuntimeError("Embedder provider service unavailable")
        self.calls.append(chunks)
        for chunk in chunks:
            chunk.embedding = [0.1] * 1536
        return chunks


class FakeRepository:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.saved_docs = []
        self.inserted_chunks = []
        self.status_updates = []
        self.generated_id = uuid4()

    async def save_document_with_chunks(
        self, filename, source_type, chunks, metadata=None, pipeline_id=None, status="completed"
    ):
        if self.should_fail:
            raise RuntimeError("Database connection failure")
        self.saved_docs.append(
            {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
                "metadata": metadata,
                "pipeline_id": pipeline_id,
                "status": status,
            }
        )
        return self.generated_id, len(chunks)

    async def insert_chunks_for_document(self, document_id, chunks):
        if self.should_fail:
            raise RuntimeError("Database chunk insertion failure")
        self.inserted_chunks.append((document_id, chunks))
        return len(chunks)

    async def update_document_status(self, document_id, status):
        self.status_updates.append((document_id, status))


@pytest.fixture
def telemetry_exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(processor)
    else:
        new_provider = TracerProvider()
        new_provider.add_span_processor(processor)
        trace._TRACER_PROVIDER = new_provider
    yield exporter
    exporter.clear()



@pytest.mark.asyncio
async def test_successful_ingestion_job_orchestration(telemetry_exporter, caplog):
    """Worker coordinates pipeline, emits span with attributes, logs completion, and completes."""
    dispatcher = FakeDispatcher()
    chunker = FakeChunker()
    embedder = FakeEmbedder()
    repo = FakeRepository()

    ctx = {
        "dispatcher": dispatcher,
        "chunker": chunker,
        "embedder": embedder,
        "repository": repo,
        "db_pool": object(),
    }

    pipeline_id = uuid4()
    with caplog.at_level(logging.INFO, logger="neuroflow.ingestion"):
        result = await process_ingestion_job(
            ctx=ctx,
            source="https://example.com/docs/annual_report.pdf",
            metadata={"department": "finance"},
            pipeline_id=pipeline_id,
        )

    # 1. Output result
    assert result["status"] == "completed"
    assert result["document_id"] == str(repo.generated_id)
    assert result["chunk_count"] == 2

    # 2. OpenTelemetry Span verification
    spans = telemetry_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ingestion.process"
    assert span.attributes["document_id"] == str(repo.generated_id)
    assert span.attributes["source_type"] == "url"
    assert span.attributes["page_count"] == 2
    assert span.attributes["chunk_count"] == 2
    assert span.attributes["embedding_calls"] == 1

    # 3. Structured JSON logging verification
    log_records = [r for r in caplog.records if r.name == "neuroflow.ingestion"]
    assert len(log_records) >= 1
    parsed_log = json.loads(log_records[-1].message)
    assert parsed_log["event"] == "ingestion_complete"
    assert parsed_log["document_id"] == str(repo.generated_id)
    assert parsed_log["chunks"] == 2
    assert parsed_log["tokens"] == 20  # 2 chunks * 10 tokens
    assert isinstance(parsed_log["duration_ms"], (int, float))
    assert parsed_log["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_pre_existing_document_status_lifecycle():
    """Pre-created document transitions: queued -> processing -> completed."""
    doc_id = uuid4()
    dispatcher = FakeDispatcher()
    chunker = FakeChunker()
    embedder = FakeEmbedder()
    repo = FakeRepository()

    ctx = {
        "dispatcher": dispatcher,
        "chunker": chunker,
        "embedder": embedder,
        "repository": repo,
        "db_pool": object(),
    }

    result = await process_ingestion_job(
        ctx=ctx,
        source="file.csv",
        document_id=doc_id,
    )

    assert result["document_id"] == str(doc_id)
    assert result["status"] == "completed"
    assert repo.status_updates == [
        (doc_id, "processing"),
        (doc_id, "completed"),
    ]


@pytest.mark.asyncio
async def test_failed_status_lifecycle_and_span_error(telemetry_exporter):
    """When processing fails, status becomes 'failed', exception recorded on span, and re-raised."""
    doc_id = uuid4()
    dispatcher = FakeDispatcher(should_fail=True)
    repo = FakeRepository()

    ctx = {
        "dispatcher": dispatcher,
        "repository": repo,
        "db_pool": object(),
    }

    with pytest.raises(ValueError, match="Extractor failed"):
        await process_ingestion_job(
            ctx=ctx,
            source="faulty.pdf",
            document_id=doc_id,
        )

    # Verify status updated to 'processing' then 'failed'
    assert repo.status_updates == [
        (doc_id, "processing"),
        (doc_id, "failed"),
    ]

    # Verify OpenTelemetry recorded the exception
    spans = telemetry_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "ingestion.process"
    assert span.status.status_code == trace.StatusCode.ERROR
    assert len(span.events) >= 1
    assert span.events[0].name == "exception"


@pytest.mark.asyncio
async def test_async_dispatcher_supported():
    """Worker handles async extractors/dispatchers transparently."""
    dispatcher = FakeAsyncDispatcher()
    chunker = FakeChunker()
    embedder = FakeEmbedder()
    repo = FakeRepository()

    ctx = {
        "dispatcher": dispatcher,
        "chunker": chunker,
        "embedder": embedder,
        "repository": repo,
        "db_pool": object(),
    }

    result = await process_ingestion_job(
        ctx=ctx,
        source="document.docx",
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] == 1
    assert repo.saved_docs[0]["source_type"] == "docx"


@pytest.mark.asyncio
async def test_empty_source_raises_value_error():
    """Empty or whitespace source raises ValueError."""
    ctx = {"db_pool": object()}
    with pytest.raises(ValueError, match="Ingestion job source cannot be empty"):
        await process_ingestion_job(ctx=ctx, source="   ")


@pytest.mark.asyncio
async def test_missing_db_pool_raises_runtime_error():
    """Missing db_pool raises clear error."""
    ctx = {
        "dispatcher": FakeDispatcher(),
        "chunker": FakeChunker(),
        "embedder": FakeEmbedder(),
    }
    with pytest.raises(RuntimeError, match="Database connection pool is not available"):
        await process_ingestion_job(ctx=ctx, source="file.pdf")
