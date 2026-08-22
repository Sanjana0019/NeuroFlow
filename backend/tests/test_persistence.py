import json
from uuid import UUID, uuid4

import pytest

from backend.db.documents import (
    DocumentRepository,
    compute_content_hash,
    format_vector,
)
from pipelines.ingestion.models import Chunk


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        self.connection.in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connection.in_transaction = False
        if exc_type is not None:
            self.connection.rolled_back = True
        else:
            self.connection.committed = True
        return False


class FakeConnection:
    def __init__(self):
        self.in_transaction = False
        self.committed = False
        self.rolled_back = False
        self.fetchval_calls = []
        self.executemany_calls = []
        self.execute_calls = []
        self.should_fail_executemany = False
        self.generated_doc_id = uuid4()

    def transaction(self):
        return FakeTransaction(self)

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return self.generated_doc_id

    async def executemany(self, query, args):
        if self.should_fail_executemany:
            raise RuntimeError("Database connection lost during chunk insertion")
        self.executemany_calls.append((query, args))

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))


class FakePool:
    def __init__(self, connection: FakeConnection):
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


def test_deterministic_content_hash():
    """Verify SHA-256 content hashing is deterministic across str and bytes."""
    text = "NeuroFlow document content test"
    hash1 = compute_content_hash(text)
    hash2 = compute_content_hash(text.encode("utf-8"))
    assert hash1 == hash2
    assert len(hash1) == 64
    assert hash1 == compute_content_hash(text)


def test_format_vector():
    """Verify vector formatting for pgvector string representation."""
    assert format_vector(None) is None
    assert format_vector([0.1, 0.25, 0.5]) == "[0.1,0.25,0.5]"


@pytest.mark.asyncio
async def test_save_document_with_chunks_success():
    """Document and chunks are inserted atomically with correct fields."""
    conn = FakeConnection()
    pool = FakePool(conn)
    repo = DocumentRepository(pool)

    chunks = [
        Chunk(
            content="First chunk content",
            chunk_index=0,
            token_count=3,
            page_number=1,
            metadata={"heading": "Intro"},
            embedding=[0.01] * 1536,
        ),
        Chunk(
            content="Second chunk content",
            chunk_index=1,
            token_count=3,
            page_number=2,
            metadata={"heading": "Details"},
            embedding=[0.02] * 1536,
        ),
    ]

    pipeline_id = uuid4()
    doc_id, count = await repo.save_document_with_chunks(
        filename="report.pdf",
        source_type="pdf",
        chunks=chunks,
        metadata={"category": "finance"},
        pipeline_id=pipeline_id,
        status="completed",
    )

    assert doc_id == conn.generated_doc_id
    assert count == 2
    assert conn.committed is True
    assert conn.rolled_back is False

    # Check document insertion query and parameters
    assert len(conn.fetchval_calls) == 1
    doc_query, doc_args = conn.fetchval_calls[0]
    assert "INSERT INTO documents" in doc_query
    assert doc_args[0] == "report.pdf"
    assert doc_args[1] == "pdf"
    assert len(doc_args[2]) == 64  # SHA-256 hash
    assert json.loads(doc_args[3]) == {"category": "finance"}
    assert doc_args[4] == pipeline_id
    assert doc_args[5] == "completed"
    assert doc_args[6] == 2  # chunk_count

    # Check chunks insertion query and parameters
    assert len(conn.executemany_calls) == 1
    chunks_query, chunk_records = conn.executemany_calls[0]
    assert "INSERT INTO chunks" in chunks_query
    assert len(chunk_records) == 2

    # Chunk 0
    assert chunk_records[0][0] == doc_id
    assert chunk_records[0][1] == "First chunk content"
    assert chunk_records[0][2].startswith("[0.01,0.01")
    assert chunk_records[0][3] == 0  # chunk_index
    assert chunk_records[0][4] == 3  # token_count
    assert json.loads(chunk_records[0][5]) == {"heading": "Intro"}

    # Chunk 1
    assert chunk_records[1][0] == doc_id
    assert chunk_records[1][1] == "Second chunk content"
    assert chunk_records[1][2].startswith("[0.02,0.02")
    assert chunk_records[1][3] == 1  # chunk_index
    assert chunk_records[1][4] == 3  # token_count
    assert json.loads(chunk_records[1][5]) == {"heading": "Details"}


@pytest.mark.asyncio
async def test_save_document_empty_chunks():
    """Document can be saved with an empty chunk list without failing."""
    conn = FakeConnection()
    pool = FakePool(conn)
    repo = DocumentRepository(pool)

    doc_id, count = await repo.save_document_with_chunks(
        filename="empty.txt",
        source_type="text",
        chunks=[],
        raw_content="",
    )

    assert doc_id == conn.generated_doc_id
    assert count == 0
    assert conn.committed is True
    assert len(conn.executemany_calls) == 0


@pytest.mark.asyncio
async def test_save_document_transaction_rollback_on_error():
    """If chunk insertion fails, transaction is rolled back and error is raised."""
    conn = FakeConnection()
    conn.should_fail_executemany = True
    pool = FakePool(conn)
    repo = DocumentRepository(pool)

    chunks = [
        Chunk(
            content="Some content",
            chunk_index=0,
            token_count=2,
            embedding=[0.1] * 1536,
        )
    ]

    with pytest.raises(RuntimeError, match="Database connection lost"):
        await repo.save_document_with_chunks(
            filename="error_doc.pdf",
            source_type="pdf",
            chunks=chunks,
        )

    assert conn.rolled_back is True
    assert conn.committed is False


@pytest.mark.asyncio
async def test_insert_chunks_for_existing_document():
    """Chunks can be added to an existing document and increment chunk_count."""
    conn = FakeConnection()
    pool = FakePool(conn)
    repo = DocumentRepository(pool)

    doc_id = uuid4()
    chunks = [
        Chunk(
            content="Follow up chunk",
            chunk_index=0,
            token_count=3,
            embedding=[0.05] * 1536,
        )
    ]

    count = await repo.insert_chunks_for_document(doc_id, chunks)

    assert count == 1
    assert conn.committed is True
    assert len(conn.executemany_calls) == 1
    assert len(conn.execute_calls) == 1
    assert "UPDATE documents" in conn.execute_calls[0][0]
    assert conn.execute_calls[0][1] == (1, doc_id)


@pytest.mark.asyncio
async def test_update_document_status():
    """Document status can be updated directly."""
    conn = FakeConnection()
    pool = FakePool(conn)
    repo = DocumentRepository(pool)

    doc_id = uuid4()
    await repo.update_document_status(doc_id, "processing")

    assert len(conn.execute_calls) == 1
    assert "UPDATE documents" in conn.execute_calls[0][0]
    assert conn.execute_calls[0][1] == ("processing", doc_id)

