import hashlib
import json
from typing import Any
from uuid import UUID

import asyncpg

from pipelines.ingestion.models import Chunk


def compute_content_hash(content: str | bytes) -> str:
    """Compute a deterministic SHA-256 hash for document content or source string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def format_vector(embedding: list[float] | None) -> str | None:
    """Format a Python list of floats as a PostgreSQL pgvector literal."""
    if embedding is None:
        return None
    return f"[{','.join(str(val) for val in embedding)}]"


class DocumentRepository:
    """Handles PostgreSQL persistence for ingested documents and their vector chunks."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def save_document_with_chunks(
        self,
        filename: str,
        source_type: str,
        chunks: list[Chunk],
        content_hash: str | None = None,
        raw_content: str | bytes | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_id: UUID | str | None = None,
        status: str = "completed",
    ) -> tuple[UUID, int]:
        """Insert a document and all its chunks atomically in a single transaction."""
        if content_hash is None:
            if raw_content is not None:
                content_hash = compute_content_hash(raw_content)
            elif chunks:
                combined_text = "".join(c.content for c in chunks)
                content_hash = compute_content_hash(combined_text)
            else:
                content_hash = compute_content_hash(filename)

        doc_metadata_json = json.dumps(metadata or {})
        pipeline_uuid = UUID(str(pipeline_id)) if pipeline_id else None

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Insert document record
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
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    filename,
                    source_type,
                    content_hash,
                    doc_metadata_json,
                    pipeline_uuid,
                    status,
                    len(chunks),
                )

                # 2. Insert chunk records if any
                if chunks:
                    chunk_records = [
                        (
                            document_id,
                            chunk.content,
                            format_vector(chunk.embedding),
                            chunk.chunk_index,
                            chunk.token_count,
                            json.dumps(chunk.metadata or {}),
                        )
                        for chunk in chunks
                    ]

                    await conn.executemany(
                        """
                        INSERT INTO chunks (
                            document_id,
                            content,
                            embedding,
                            chunk_index,
                            token_count,
                            metadata
                        )
                        VALUES ($1, $2, $3::vector, $4, $5, $6)
                        """,
                        chunk_records,
                    )

                return document_id, len(chunks)

    async def insert_chunks_for_document(
        self,
        document_id: UUID | str,
        chunks: list[Chunk],
    ) -> int:
        """Insert chunks for an existing document record within a transaction."""
        if not chunks:
            return 0

        doc_uuid = UUID(str(document_id))
        chunk_records = [
            (
                doc_uuid,
                chunk.content,
                format_vector(chunk.embedding),
                chunk.chunk_index,
                chunk.token_count,
                json.dumps(chunk.metadata or {}),
            )
            for chunk in chunks
        ]

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO chunks (
                        document_id,
                        content,
                        embedding,
                        chunk_index,
                        token_count,
                        metadata
                    )
                    VALUES ($1, $2, $3::vector, $4, $5, $6)
                    """,
                    chunk_records,
                )
                await conn.execute(
                    """
                    UPDATE documents
                    SET chunk_count = COALESCE(chunk_count, 0) + $1
                    WHERE id = $2
                    """,
                    len(chunks),
                    doc_uuid,
                )

        return len(chunks)

    async def update_document_status(
        self,
        document_id: UUID | str,
        status: str,
    ) -> None:
        """Update the processing status of a document."""
        doc_uuid = UUID(str(document_id))
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET status = $1
                WHERE id = $2
                """,
                status,
                doc_uuid,
            )

    async def get_document(
        self,
        document_id: UUID | str,
    ) -> dict[str, Any] | None:
        """Fetch document record by ID."""
        doc_uuid = UUID(str(document_id))
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, filename, source_type, content_hash, metadata, pipeline_id, status, chunk_count, created_at
                FROM documents
                WHERE id = $1
                """,
                doc_uuid,
            )
            if row is None:
                return None
            return dict(row)

