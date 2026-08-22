import io
import json
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.ingest import router as ingest_router


class FakeARQRedis:
    def __init__(self):
        self.enqueued_jobs = []

    async def enqueue_job(self, function_name, **kwargs):
        self.enqueued_jobs.append({"function": function_name, "args": kwargs})
        return "job-123"


class FakeDBConnection:
    def __init__(self, documents=None):
        self.documents = documents or {}  # content_hash -> dict, and id -> dict
        self.inserted_docs = []

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())
        if "WHERE content_hash = $1" in normalized_query:
            content_hash = args[0]
            return self.documents.get(content_hash)
        elif "WHERE id = $1" in normalized_query:
            doc_id = args[0]
            # Match by either UUID or string
            return self.documents.get(doc_id) or self.documents.get(UUID(str(doc_id))) or self.documents.get(str(doc_id))
        return None

    async def fetchval(self, query, *args):
        normalized_query = " ".join(query.split())
        if "INSERT INTO documents" in normalized_query:
            filename, source_type, content_hash, meta_str, pipeline_id = args
            doc_id = uuid4()
            doc_data = {
                "id": doc_id,
                "filename": filename,
                "source_type": source_type,
                "content_hash": content_hash,
                "metadata": meta_str,
                "pipeline_id": pipeline_id,
                "status": "queued",
                "chunk_count": 0,
            }
            self.documents[content_hash] = doc_data
            self.documents[doc_id] = doc_data
            self.documents[str(doc_id)] = doc_data
            self.inserted_docs.append(doc_data)
            return doc_id
        return None

    async def execute(self, query, *args):
        pass


class FakeDBPool:
    def __init__(self, connection):
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


@pytest.fixture
def api_client():
    app = FastAPI()
    app.include_router(ingest_router)

    conn = FakeDBConnection()
    pool = FakeDBPool(conn)
    arq_redis = FakeARQRedis()

    app.state.db_pool = pool
    app.state.arq_redis = arq_redis

    client = TestClient(app)
    return client, conn, arq_redis


def test_ingest_json_url_success(api_client):
    """POST /ingest with JSON URL creates a queued document and enqueues an ARQ job."""
    client, conn, arq_redis = api_client

    response = client.post(
        "/ingest",
        json={"url": "https://example.com/research.pdf", "metadata": {"category": "ai"}},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "queued"
    assert data["duplicate"] is False

    # Check database insertion
    assert len(conn.inserted_docs) == 1
    assert conn.inserted_docs[0]["source_type"] == "url"

    # Check ARQ job enqueue
    assert len(arq_redis.enqueued_jobs) == 1
    job = arq_redis.enqueued_jobs[0]
    assert job["function"] == "process_ingestion_job"
    assert job["args"]["source"] == "https://example.com/research.pdf"
    assert job["args"]["source_type"] == "url"
    assert job["args"]["document_id"] == data["document_id"]


def test_ingest_multipart_file_success(api_client):
    """POST /ingest with multipart/form-data creates a queued document and enqueues job."""
    client, conn, arq_redis = api_client

    file_content = b"Col1,Col2\nVal1,Val2\n"
    response = client.post(
        "/ingest",
        files={"file": ("dataset.csv", io.BytesIO(file_content), "text/csv")},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "queued"
    assert data["duplicate"] is False

    assert len(conn.inserted_docs) == 1
    assert conn.inserted_docs[0]["filename"] == "dataset.csv"
    assert conn.inserted_docs[0]["source_type"] == "csv"

    assert len(arq_redis.enqueued_jobs) == 1
    job = arq_redis.enqueued_jobs[0]
    assert job["args"]["source_type"] == "csv"
    assert job["args"]["filename"] == "dataset.csv"


def test_duplicate_upload_deduplication(api_client):
    """Submitting the identical file twice returns duplicate=True on the second call without re-queueing."""
    client, conn, arq_redis = api_client

    file_content = b"PDF dummy binary content for deduplication test"

    # 1. First upload
    res1 = client.post(
        "/ingest",
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
    )
    assert res1.status_code == 202
    data1 = res1.json()
    assert data1["duplicate"] is False
    doc_id1 = data1["document_id"]
    assert len(arq_redis.enqueued_jobs) == 1

    # 2. Second upload with identical bytes
    res2 = client.post(
        "/ingest",
        files={"file": ("test_renamed.pdf", io.BytesIO(file_content), "application/pdf")},
    )
    assert res2.status_code == 202
    data2 = res2.json()
    assert data2["duplicate"] is True
    assert data2["document_id"] == doc_id1
    # Ensure no second job was enqueued
    assert len(arq_redis.enqueued_jobs) == 1


def test_unsupported_file_type_rejected(api_client):
    """Unsupported file extensions are rejected with 400 Bad Request."""
    client, conn, arq_redis = api_client

    response = client.post(
        "/ingest",
        files={"file": ("malicious.exe", io.BytesIO(b"executable content"), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert len(arq_redis.enqueued_jobs) == 0


def test_missing_input_rejected(api_client):
    """Empty JSON or missing file returns 400 Bad Request."""
    client, conn, arq_redis = api_client

    res1 = client.post("/ingest", json={})
    assert res1.status_code == 400

    res2 = client.post("/ingest", files={})
    assert res2.status_code == 400


def test_get_document_status_success(api_client):
    """GET /documents/{document_id} retrieves document status, chunk count, and metadata."""
    client, conn, arq_redis = api_client

    doc_id = uuid4()
    conn.documents[doc_id] = {
        "id": doc_id,
        "filename": "annual.docx",
        "source_type": "docx",
        "content_hash": "abc123hash",
        "metadata": json.dumps({"author": "Sanjana"}),
        "pipeline_id": None,
        "status": "completed",
        "chunk_count": 8,
    }

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "completed"
    assert data["chunk_count"] == 8
    assert data["metadata"] == {"author": "Sanjana"}


def test_get_document_status_not_found(api_client):
    """GET /documents/{missing_id} returns 404 Not Found."""
    client, conn, arq_redis = api_client

    missing_id = uuid4()
    response = client.get(f"/documents/{missing_id}")
    assert response.status_code == 404
    assert f"Document '{missing_id}' not found" in response.json()["detail"]


@pytest.mark.parametrize("status", ["queued", "processing", "completed", "failed"])
def test_get_document_all_status_lifecycle_values(api_client, status):
    """GET /documents/{id} correctly reports all lifecycle status values."""
    client, conn, arq_redis = api_client

    doc_id = uuid4()
    conn.documents[doc_id] = {
        "id": doc_id,
        "filename": "test.txt",
        "source_type": "text",
        "content_hash": f"hash_{status}",
        "metadata": "{}",
        "pipeline_id": None,
        "status": status,
        "chunk_count": 5 if status == "completed" else 0,
    }

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["status"] == status
