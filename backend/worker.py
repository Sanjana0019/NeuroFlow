import inspect
import json
import logging
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from arq.connections import RedisSettings
from opentelemetry import trace
from redis.asyncio import Redis

try:
    from backend.config import settings
    from backend.db.documents import DocumentRepository
    from backend.db.pool import create_pool
    from backend.providers.anthropic_provider import AnthropicProvider
    from backend.providers.client import NeuroFlowClient, set_client
    from backend.providers.openai_provider import OpenAIProvider
except ImportError:
    from config import settings
    from db.documents import DocumentRepository
    from db.pool import create_pool
    from providers.anthropic_provider import AnthropicProvider
    from providers.client import NeuroFlowClient, set_client
    from providers.openai_provider import OpenAIProvider

from pipelines.ingestion.chunker import Chunker
from pipelines.ingestion.dispatcher import IngestionDispatcher
from pipelines.ingestion.embedder import ChunkEmbedder
from backend.resilience.timeout_manager import TimeoutManager

logger = logging.getLogger("neuroflow.ingestion")
tracer = trace.get_tracer("neuroflow.ingestion")

EXTENSION_TO_SOURCE_TYPE = {
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


def _infer_source_type(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and bool(parsed.netloc):
        return "url"
    ext = Path(source).suffix.lower()
    return EXTENSION_TO_SOURCE_TYPE.get(ext, "text")


def _infer_filename(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and bool(parsed.netloc):
        return source.rstrip("/").split("/")[-1] or "remote-document"
    return Path(source).name or "document"


async def process_ingestion_job(
    ctx: dict[str, Any],
    source: str,
    source_type: str | None = None,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    pipeline_id: UUID | str | None = None,
    document_id: UUID | str | None = None,
) -> dict[str, Any]:
    """Orchestrate ingestion with OpenTelemetry tracing, status lifecycle, and structured logging."""
    if not source or not source.strip():
        raise ValueError("Ingestion job source cannot be empty")

    source = source.strip()
    inferred_source_type = source_type or _infer_source_type(source)
    inferred_filename = filename or _infer_filename(source)
    job_metadata = metadata or {}
    start_time = time.perf_counter()

    persisted_doc_id = document_id
    db_pool = ctx.get("db_pool")
    if db_pool is None:
        raise RuntimeError("Database connection pool is not available in worker context")

    repository = ctx.get("repository") or DocumentRepository(db_pool)

    with tracer.start_as_current_span("ingestion.process") as span:
        span.set_attribute("source_type", inferred_source_type)
        if persisted_doc_id is not None:
            span.set_attribute("document_id", str(persisted_doc_id))

        try:
            # Transition to 'processing' status if document was pre-created (e.g. queued)
            if persisted_doc_id is not None:
                await repository.update_document_status(persisted_doc_id, "processing")

            # 1. Extraction via Unified Ingestion Dispatcher with TimeoutManager
            dispatcher = ctx.get("dispatcher") or IngestionDispatcher()
            timeout_mgr = ctx.get("timeout_manager") or TimeoutManager(redis=ctx.get("redis"))
            task_type = "url_fetch" if inferred_source_type == "url" else "file_extraction"

            async def _extract_call():
                res = dispatcher.dispatch(source)
                if inspect.iscoroutine(res):
                    return await res
                return res

            extracted_pages = await timeout_mgr.execute(task_type, _extract_call())

            span.set_attribute("page_count", len(extracted_pages))

            # 2. Structure-aware chunking
            chunker = ctx.get("chunker") or Chunker()
            chunks = chunker.chunk(extracted_pages, document_id=persisted_doc_id)
            span.set_attribute("chunk_count", len(chunks))

            # 3. Vector Embedding
            embedder = ctx.get("embedder") or ChunkEmbedder(client=ctx.get("neuroflow_client"))
            embedding_calls = math.ceil(len(chunks) / 100) if chunks else 0
            span.set_attribute("embedding_calls", embedding_calls)

            embedded_chunks = await embedder.embed_chunks(chunks)

            # 4. PostgreSQL Persistence & Status Update
            if persisted_doc_id is not None:
                chunk_count = await repository.insert_chunks_for_document(
                    document_id=persisted_doc_id,
                    chunks=embedded_chunks,
                )
                await repository.update_document_status(persisted_doc_id, "completed")
            else:
                persisted_doc_id, chunk_count = await repository.save_document_with_chunks(
                    filename=inferred_filename,
                    source_type=inferred_source_type,
                    chunks=embedded_chunks,
                    metadata=job_metadata,
                    pipeline_id=pipeline_id,
                    status="completed",
                )
                span.set_attribute("document_id", str(persisted_doc_id))

            duration_ms = (time.perf_counter() - start_time) * 1000
            total_tokens = sum(c.token_count for c in embedded_chunks)

            # Structured JSON completion log
            completion_log = {
                "event": "ingestion_complete",
                "document_id": str(persisted_doc_id),
                "duration_ms": round(duration_ms, 2),
                "chunks": chunk_count,
                "tokens": total_tokens,
            }
            logger.info(json.dumps(completion_log))

            return {
                "document_id": str(persisted_doc_id),
                "chunk_count": chunk_count,
                "status": "completed",
            }

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))

            if persisted_doc_id is not None:
                try:
                    await repository.update_document_status(persisted_doc_id, "failed")
                except Exception:
                    pass
            raise


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize shared database, Redis, and client resources on worker startup."""
    ctx["db_pool"] = await create_pool()

    ctx["redis"] = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        decode_responses=True,
    )

    providers = {}
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(api_key=settings.openai_api_key)
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)

    client = NeuroFlowClient(
        redis=ctx["redis"],
        providers=providers,
    )
    ctx["neuroflow_client"] = client
    set_client(client)


async def evaluate_pipeline_run(ctx: dict[str, Any], run_id: str) -> None:
    """ARQ background job to run EvaluationJudge on a completed pipeline run."""
    db_pool = ctx.get("db_pool")
    client = ctx.get("neuroflow_client")
    if not db_pool:
        return

    from uuid import UUID
    from evaluation.judge import EvaluationJudge

    run_uuid = UUID(str(run_id))
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, query, generation, retrieved_chunk_ids
            FROM pipeline_runs
            WHERE id = $1
            """,
            run_uuid,
        )
        if not row:
            return

        query_text = row["query"]
        generation_text = row["generation"] or ""
        chunk_ids = row["retrieved_chunk_ids"] or []

        chunks = []
        if chunk_ids:
            chunk_rows = await conn.fetch(
                "SELECT content FROM chunks WHERE id = ANY($1::uuid[])",
                chunk_ids,
            )
            chunks = [r["content"] for r in chunk_rows]

    judge = EvaluationJudge(client=client, db_pool=db_pool)
    await judge.evaluate_run(
        run_id=run_uuid,
        query=query_text,
        answer=generation_text,
        context=chunks,
    )


async def poll_finetune_job(ctx: dict[str, Any], job_id: str) -> None:
    """ARQ background job to poll fine-tuning job status from OpenAI and register model on completion."""
    db_pool = ctx.get("db_pool")
    redis_client = ctx.get("redis")
    if not db_pool:
        return

    from pipelines.finetuning.job_manager import FineTuningJobManager
    manager = FineTuningJobManager()
    await manager.poll_job_status(
        job_id=job_id,
        db_pool=db_pool,
        redis_client=redis_client,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    """Close active resources on worker shutdown."""
    if "redis" in ctx:
        await ctx["redis"].aclose()
    if "db_pool" in ctx:
        await ctx["db_pool"].close()


class WorkerSettings:
    """ARQ Worker configuration settings."""

    functions = [process_ingestion_job, evaluate_pipeline_run, poll_finetune_job]
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
    )
    on_startup = startup
    on_shutdown = shutdown