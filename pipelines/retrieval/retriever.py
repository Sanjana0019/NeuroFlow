import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from backend.db.documents import format_vector
from pipelines.retrieval.models import ProcessedQuery, RetrievalResult

logger = logging.getLogger("neuroflow.retrieval.retriever")


class Retriever:
    """Performs parallel hybrid retrieval across Dense (pgvector), Sparse (ts_rank_cd), and Metadata channels."""

    def __init__(self, db_pool, embedder):
        self.db_pool = db_pool
        self.embedder = embedder

    async def _dense_search_single(
        self,
        query_text: str,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """Dense search for a single query embedding."""
        if not query_text.strip():
            return []

        try:
            embeddings = await self.embedder.embed([query_text])
        except Exception as exc:
            logger.warning("Dense embedding generation failed (%s). Continuing with sparse retrieval.", exc)
            return []

        if not embeddings or not embeddings[0]:
            return []

        vector_str = format_vector(embeddings[0])

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata,
                       d.filename,
                       1.0 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
                """,
                vector_str,
                k,
            )

        results = []
        for rank, r in enumerate(rows, start=1):
            meta = r["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            results.append(
                RetrievalResult(
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    score=float(r["score"] or 0.0),
                    rank=rank,
                    source="dense",
                    filename=r["filename"],
                    page_number=meta.get("page_number") if isinstance(meta, dict) else None,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return results

    async def dense_retrieval(
        self,
        query: str,
        expanded_queries: list[str] | None = None,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """Dense retrieval searching for the main query plus any expanded alternatives."""
        all_queries = [query]
        if expanded_queries:
            all_queries.extend([q for q in expanded_queries if q.strip() and q != query])

        tasks = [self._dense_search_single(q, k=k) for q in all_queries]
        multi_results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_chunks: dict[str, RetrievalResult] = {}
        for res in multi_results:
            if isinstance(res, Exception) or not res:
                continue
            for item in res:
                key = str(item.chunk_id)
                if key not in seen_chunks or item.score > seen_chunks[key].score:
                    seen_chunks[key] = item

        sorted_dense = sorted(seen_chunks.values(), key=lambda r: r.score, reverse=True)
        # Update ranks
        for idx, item in enumerate(sorted_dense, start=1):
            item.rank = idx
        return sorted_dense[:k]

    async def sparse_retrieval(
        self,
        query: str,
        k: int = 20,
    ) -> list[RetrievalResult]:
        """Sparse lexical retrieval using PostgreSQL Full-Text Search with fallback ranking."""
        if not query.strip():
            return []

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata,
                       d.filename,
                       ts_rank_cd(to_tsvector('english', c.content), websearch_to_tsquery('english', $1)) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ websearch_to_tsquery('english', $1)
                   OR to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                k,
            )

            if not rows:
                rows = await conn.fetch(
                    """
                    SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata,
                           d.filename,
                           ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', $1)) AS score
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY score DESC, c.chunk_index ASC
                    LIMIT $2
                    """,
                    query,
                    k,
                )

        results = []
        for rank, r in enumerate(rows, start=1):
            meta = r["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            results.append(
                RetrievalResult(
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    score=float(r["score"] or 0.0),
                    rank=rank,
                    source="sparse",
                    filename=r["filename"],
                    page_number=meta.get("page_number") if isinstance(meta, dict) else None,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return results

    async def metadata_retrieval(
        self,
        query: str,
        filters: dict[str, Any],
        k: int = 20,
    ) -> list[RetrievalResult]:
        """Metadata-constrained vector retrieval matching JSONB filters."""
        if not filters or not query.strip():
            return []

        filter_json = json.dumps(filters)
        embeddings = await self.embedder.embed([query])
        if not embeddings or not embeddings[0]:
            return []

        vector_str = format_vector(embeddings[0])

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata,
                       d.filename,
                       1.0 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE (c.metadata @> $2::jsonb OR d.metadata @> $2::jsonb)
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3
                """,
                vector_str,
                filter_json,
                k,
            )

        results = []
        for rank, r in enumerate(rows, start=1):
            meta = r["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            results.append(
                RetrievalResult(
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    score=float(r["score"] or 0.0),
                    rank=rank,
                    source="metadata",
                    filename=r["filename"],
                    page_number=meta.get("page_number") if isinstance(meta, dict) else None,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return results

    async def retrieve_parallel(
        self,
        query: str | ProcessedQuery,
        k: int = 20,
        metadata_filters: dict[str, Any] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Concurrently execute Dense, Sparse, and Metadata retrieval using asyncio.gather."""
        if isinstance(query, ProcessedQuery):
            main_query = query.original_query
            expansions = query.expanded_queries
            filters = metadata_filters or query.metadata_filters
        else:
            main_query = query
            expansions = []
            filters = metadata_filters or {}

        dense_task = self.dense_retrieval(main_query, expanded_queries=expansions, k=k)
        sparse_task = self.sparse_retrieval(main_query, k=k)
        metadata_task = self.metadata_retrieval(main_query, filters=filters, k=k) if filters else asyncio.sleep(0, result=[])

        dense_res, sparse_res, meta_res = await asyncio.gather(
            dense_task,
            sparse_task,
            metadata_task,
            return_exceptions=True,
        )

        return {
            "dense": dense_res if isinstance(dense_res, list) else [],
            "sparse": sparse_res if isinstance(sparse_res, list) else [],
            "metadata": meta_res if isinstance(meta_res, list) else [],
        }
