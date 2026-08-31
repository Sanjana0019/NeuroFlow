from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

logger = logging.getLogger("neuroflow.api.ingest")

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.db.documents import DocumentRepository
from backend.resilience.backpressure import check_ingestion_backpressure
from backend.resilience.rate_limiter import EndpointRateLimiter

router = APIRouter(tags=["Ingestion"])

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".txt": "text",
    ".pptx": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
}


class IngestResponse(BaseModel):
    document_id: UUID
    status: str
    duplicate: bool = False
    warning: str | None = None
    estimated_wait_minutes: int | None = None
    queue_depth: int | None = None


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: str
    chunk_count: int = 0
    metadata: dict[str, Any] = {}


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(request: Request):
    """Enqueue document ingestion job from either multipart file upload or JSON URL."""
    # 1. API Rate Limiting (10 requests/hour/IP)
    redis_client = getattr(request.app.state, "redis", None) or getattr(request.app.state, "arq_redis", None)
    client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "127.0.0.1")
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    allowed, retry_after = await EndpointRateLimiter.check_rate_limit(
        redis=redis_client,
        client_ip=client_ip,
        endpoint="ingest",
        max_requests=10,
        window_seconds=3600,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for ingestion (10 requests/hour/IP)",
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Ingestion Backpressure Check
    bp_result = await check_ingestion_backpressure(redis_client)
    if bp_result["action"] == "reject":
        return JSONResponse(
            status_code=503,
            content=bp_result["payload"],
        )

    content_type = request.headers.get("content-type", "")
    db_pool = request.app.state.db_pool

    source = ""
    source_type = ""
    filename = ""
    content_hash = ""
    metadata: dict[str, Any] = {}
    pipeline_id = None

    # Case A: JSON Body (URL ingestion)
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        url = body.get("url") or body.get("source")
        if not url or not str(url).strip():
            raise HTTPException(status_code=400, detail="Missing 'url' field in JSON request")

        url_str = str(url).strip()
        parsed = urlparse(url_str)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail=f"Invalid URL: {url_str}")

        source = url_str
        source_type = "url"
        filename = url_str.rstrip("/").split("/")[-1] or "remote-document"
        content_hash = hashlib.sha256(url_str.encode("utf-8")).hexdigest()
        metadata = body.get("metadata") or {}
        pipeline_id = body.get("pipeline_id")

    # Case B: Multipart Form (File upload)
    elif "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file or not hasattr(uploaded_file, "read"):
            raise HTTPException(status_code=400, detail="Missing 'file' in multipart form data")

        original_filename = uploaded_file.filename or "upload"
        ext = Path(original_filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}",
            )

        source_type = SUPPORTED_EXTENSIONS[ext]
        filename = original_filename

        # Stream/chunk reading with size limit check
        sha = hashlib.sha256()
        total_size = 0
        file_chunks: list[bytes] = []

        while True:
            chunk = await uploaded_file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB",
                )
            sha.update(chunk)
            file_chunks.append(chunk)

        content_hash = sha.hexdigest()

        # Save file to local upload directory
        saved_path = UPLOAD_DIR / f"{content_hash}_{filename}"
        if not saved_path.exists():
            with open(saved_path, "wb") as f:
                for chunk in file_chunks:
                    f.write(chunk)

        source = str(saved_path.resolve())

        meta_field = form.get("metadata")
        if meta_field:
            if isinstance(meta_field, str):
                try:
                    metadata = json.loads(meta_field)
                except Exception:
                    metadata = {}
            elif isinstance(meta_field, dict):
                metadata = meta_field

        p_id = form.get("pipeline_id")
        if p_id:
            try:
                pipeline_id = UUID(str(p_id))
            except Exception:
                pipeline_id = None

    else:
        raise HTTPException(
            status_code=400,
            detail="Content-Type must be 'application/json' or 'multipart/form-data'",
        )

    # -------------------------------------------------------------
    # 2. Deduplication check
    # -------------------------------------------------------------
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, status, chunk_count
            FROM documents
            WHERE content_hash = $1
            """,
            content_hash,
        )
        if existing:
            return IngestResponse(
                document_id=existing["id"],
                status=existing["status"],
                duplicate=True,
            )

        # -------------------------------------------------------------
        # 3. Create document record with status = 'queued'
        # -------------------------------------------------------------
        try:
            document_id = await conn.fetchval(
                """
                INSERT INTO documents (
                    filename,
                    source_type,
                    content_hash,
                    metadata,
                    pipeline_id,
                    status,
                    chunk_count
                )
                VALUES ($1, $2, $3, $4, $5, 'queued', 0)
                RETURNING id
                """,
                filename,
                source_type,
                content_hash,
                json.dumps(metadata),
                pipeline_id,
            )
        except Exception as exc:
            # Race condition handling for concurrent duplicate uploads
            if "documents_content_hash_key" in str(exc) or "unique constraint" in str(exc).lower():
                existing = await conn.fetchrow(
                    "SELECT id, status FROM documents WHERE content_hash = $1",
                    content_hash,
                )
                return IngestResponse(
                    document_id=existing["id"],
                    status=existing["status"],
                    duplicate=True,
                )
            raise

    # -------------------------------------------------------------
    # 4. Enqueue ARQ job asynchronously
    # -------------------------------------------------------------
    arq_redis = getattr(request.app.state, "arq_redis", None)
    if arq_redis is not None:
        await arq_redis.enqueue_job(
            "process_ingestion_job",
            source=source,
            source_type=source_type,
            filename=filename,
            metadata=metadata,
            pipeline_id=str(pipeline_id) if pipeline_id else None,
            document_id=str(document_id),
        )

    warning = bp_result.get("warning") if bp_result.get("action") == "warn" else None
    estimated_wait = bp_result.get("estimated_wait_minutes") if bp_result.get("action") == "warn" else None

    return IngestResponse(
        document_id=document_id,
        status="queued",
        duplicate=False,
        warning=warning,
        estimated_wait_minutes=estimated_wait,
        queue_depth=bp_result.get("queue_depth"),
    )


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: UUID,
    request: Request,
):
    """Retrieve document ingestion status, chunk count, and metadata."""
    db_pool = request.app.state.db_pool
    repo = DocumentRepository(db_pool)
    doc = await repo.get_document(document_id)

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found",
        )

    raw_metadata = doc.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            doc_metadata = json.loads(raw_metadata)
        except Exception:
            doc_metadata = {}
    else:
        doc_metadata = raw_metadata or {}

    return DocumentStatusResponse(
        document_id=doc["id"],
        status=doc["status"],
        chunk_count=doc.get("chunk_count") or 0,
        metadata=doc_metadata,
    )


class DocumentItemResponse(BaseModel):
    id: UUID
    filename: str
    source_type: str
    status: str
    chunk_count: int = 0
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None


class ChunkItemResponse(BaseModel):
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int = 0
    metadata: dict[str, Any] = {}


class SimilarSearchRequest(BaseModel):
    query: str
    limit: int = 5


class SimilarChunkResponse(BaseModel):
    id: UUID
    chunk_index: int
    content: str
    similarity_score: float


@router.get("/documents", response_model=list[DocumentItemResponse])
async def list_documents(request: Request):
    """List all ingested documents with metadata, status, and chunk count."""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool is not available")

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, source_type, status, chunk_count, metadata, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        )

    results: list[DocumentItemResponse] = []
    for r in rows:
        meta_raw = r["metadata"]
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
        else:
            meta = meta_raw or {}

        results.append(
            DocumentItemResponse(
                id=r["id"],
                filename=r["filename"],
                source_type=r["source_type"],
                status=r["status"],
                chunk_count=r["chunk_count"] or 0,
                metadata=meta,
                created_at=r["created_at"],
            )
        )

    return results


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkItemResponse])
async def get_document_chunks(document_id: UUID, request: Request):
    """Retrieve all chunks belonging to a specific document."""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool is not available")

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, chunk_index, content, token_count, metadata
            FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index ASC
            """,
            document_id,
        )

    results: list[ChunkItemResponse] = []
    for r in rows:
        meta_raw = r["metadata"]
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
        else:
            meta = meta_raw or {}

        results.append(
            ChunkItemResponse(
                id=r["id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                token_count=r["token_count"] or 0,
                metadata=meta,
            )
        )

    return results


@router.post("/documents/{document_id}/similar", response_model=list[SimilarChunkResponse])
async def find_similar_chunks(
    document_id: UUID,
    body: SimilarSearchRequest,
    request: Request,
):
    """Find the most semantically similar chunks in a document for a search query using pgvector."""
    db_pool = request.app.state.db_pool
    client = getattr(request.app.state, "neuroflow_client", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool is not available")

    query_text = (body.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    # Embed search query
    embeddings = []
    if client:
        try:
            embeddings = await client.embed([query_text])
        except Exception as exc:
            logger.warning("Embed query failed in find_similar_chunks: %s", exc)

    if not embeddings or not embeddings[0]:
        # Fallback text ILIKE matching if embedder is offline
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, chunk_index, content, 0.85 AS similarity_score
                FROM chunks
                WHERE document_id = $1 AND content ILIKE $2
                LIMIT $3
                """,
                document_id,
                f"%{query_text}%",
                body.limit,
            )
            return [
                SimilarChunkResponse(
                    id=r["id"],
                    chunk_index=r["chunk_index"],
                    content=r["content"],
                    similarity_score=float(r["similarity_score"]),
                )
                for r in rows
            ]

    vector_str = f"[{','.join(str(v) for v in embeddings[0])}]"

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, chunk_index, content, (1 - (embedding <=> $1::vector)) AS similarity_score
            FROM chunks
            WHERE document_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vector_str,
            document_id,
            body.limit,
        )

    return [
        SimilarChunkResponse(
            id=r["id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            similarity_score=round(max(0.0, float(r["similarity_score"] or 0.0)), 4),
        )
        for r in rows
    ]