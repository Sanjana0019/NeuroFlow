from hashlib import sha256
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, HttpUrl


router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    source_type: str
    source: HttpUrl
    metadata: dict = {}


class IngestResponse(BaseModel):
    document_id: UUID
    status: str


@router.post("", response_model=IngestResponse, status_code=202)
async def ingest_document(
    data: IngestRequest,
    request: Request,
):
    """Create a document ingestion job."""

    if data.source_type != "url":
        raise HTTPException(
            status_code=400,
            detail="First implementation supports source_type='url' only",
        )

    db_pool = request.app.state.db_pool

    source = str(data.source)

    content_hash = sha256(source.encode("utf-8")).hexdigest()

    filename = source.rstrip("/").split("/")[-1] or "remote-document"

    async with db_pool.acquire() as conn:
        try:
            document_id = await conn.fetchval(
                """
                INSERT INTO documents (
                    filename,
                    source_type,
                    content_hash,
                    metadata,
                    status
                )
                VALUES ($1, $2, $3, $4, 'processing')
                RETURNING id
                """,
                filename,
                data.source_type,
                content_hash,
                data.metadata,
            )
        except Exception as exc:
            if "documents_content_hash_key" in str(exc):
                raise HTTPException(
                    status_code=409,
                    detail="Document has already been submitted",
                ) from exc

            raise

    return IngestResponse(
        document_id=document_id,
        status="processing",
    )