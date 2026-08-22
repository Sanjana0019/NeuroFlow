import asyncio
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pipelines.generation.citations import Citation
from pipelines.generation.generator import Generator
from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import Reranker
from pipelines.retrieval.retriever import Retriever

logger = logging.getLogger("neuroflow.api.query")

router = APIRouter(tags=["Query"])

# In-memory registry of active streaming query runs
active_stream_runs: dict[str, dict[str, Any]] = {}


class QueryRequest(BaseModel):
    query: str
    pipeline_id: UUID | None = None
    stream: bool = False


class CitationResponse(BaseModel):
    reference: str
    chunk_id: str | None = None
    document_name: str
    page_number: int | None = None
    content_preview: str
    invalid_citation: bool = False


class QueryResponse(BaseModel):
    run_id: UUID
    query: str
    generation: str
    citations: list[CitationResponse] = []
    sources: list[dict[str, Any]] = []
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: str = "gpt-4o-mini"
    status: str = "complete"


def _get_retrieval_pipeline(request: Request) -> RetrievalPipeline:
    """Instantiate RetrievalPipeline from app state resources."""
    db_pool = request.app.state.db_pool
    client = getattr(request.app.state, "neuroflow_client", None)

    retriever = Retriever(db_pool=db_pool, embedder=client)
    query_processor = QueryProcessor(client=client)
    reranker = Reranker(client=client)
    context_assembler = ContextAssembler(token_budget=4000)

    return RetrievalPipeline(
        retriever=retriever,
        query_processor=query_processor,
        reranker=reranker,
        context_assembler=context_assembler,
    )


@router.post("/query")
async def execute_query(
    body: QueryRequest,
    request: Request,
):
    """Execute grounded RAG query with either full JSON response or SSE streaming setup."""
    query_text = (body.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    db_pool = request.app.state.db_pool
    client = getattr(request.app.state, "neuroflow_client", None)
    arq_redis = getattr(request.app.state, "arq_redis", None)

    # 1. Non-Streaming: execute end-to-end and return complete JSON
    if not body.stream:
        pipeline = _get_retrieval_pipeline(request)
        chunks, assembled_context, processed_query = await pipeline.run(
            query=query_text,
            mode="full",
            top_k=5,
        )

        generator = Generator(client=client)
        output = await generator.generate(
            query=query_text,
            assembled_context=assembled_context,
            chunks_used=chunks,
            query_type=processed_query.query_type,
            pipeline_id=body.pipeline_id,
            db_pool=db_pool,
            arq_redis=arq_redis,
        )

        return QueryResponse(
            run_id=output.run_id,
            query=output.query,
            generation=output.generation,
            citations=[
                CitationResponse(
                    reference=c.reference,
                    chunk_id=str(c.chunk_id) if c.chunk_id else None,
                    document_name=c.document_name,
                    page_number=c.page_number,
                    content_preview=c.content_preview,
                    invalid_citation=c.invalid_citation,
                )
                for c in output.citations
            ],
            sources=output.sources,
            latency_ms=output.latency_ms,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            model_used=output.model_used,
            status="complete",
        )

    # 2. Streaming Setup: register active run and return run_id for SSE connection
    run_id = uuid4()
    active_stream_runs[str(run_id)] = {
        "query": query_text,
        "pipeline_id": body.pipeline_id,
        "created_at": asyncio.get_event_loop().time(),
    }

    return {
        "run_id": str(run_id),
        "status": "started",
    }


@router.get("/query/{run_id}/stream")
async def stream_query_events(
    run_id: str,
    request: Request,
):
    """Server-Sent Events (SSE) streaming endpoint emitting retrieval progress, tokens, and 15s keepalives."""
    run_info = active_stream_runs.pop(run_id, None)
    if not run_info:
        raise HTTPException(
            status_code=404,
            detail=f"Streaming run '{run_id}' not found or already consumed",
        )

    query_text = run_info["query"]
    pipeline_id = run_info.get("pipeline_id")

    db_pool = request.app.state.db_pool
    client = getattr(request.app.state, "neuroflow_client", None)
    arq_redis = getattr(request.app.state, "arq_redis", None)

    async def event_generator():
        # 1. Retrieval Start Event
        yield {"data": json.dumps({"type": "retrieval_start"})}

        # 2. Execute Task 5 Retrieval
        pipeline = _get_retrieval_pipeline(request)
        chunks, assembled_context, processed_query = await pipeline.run(
            query=query_text,
            mode="full",
            top_k=5,
        )

        sources_list = [
            {
                "source_index": idx,
                "document_id": str(c.document_id),
                "filename": c.filename,
                "page_number": c.page_number,
            }
            for idx, c in enumerate(chunks, start=1)
        ]

        # 3. Retrieval Complete Event
        yield {
            "data": json.dumps(
                {
                    "type": "retrieval_complete",
                    "chunk_count": len(chunks),
                    "sources": sources_list,
                }
            )
        }

        # 4. Stream LLM tokens with 15s keepalive support
        generator = Generator(client=client)
        stream_gen = generator.stream_generation(
            query=query_text,
            assembled_context=assembled_context,
            chunks_used=chunks,
            query_type=processed_query.query_type,
            pipeline_id=pipeline_id,
            db_pool=db_pool,
            arq_redis=arq_redis,
        )

        # Iterate generator with keepalive watchdog
        last_yield_time = asyncio.get_event_loop().time()
        KEEPALIVE_INTERVAL = 15.0

        stream_finished = False
        gen_task = None

        while not stream_finished:
            if gen_task is None:
                gen_task = asyncio.create_task(stream_gen.__anext__())

            now = asyncio.get_event_loop().time()
            time_since_yield = now - last_yield_time
            time_to_next_keepalive = max(0.1, KEEPALIVE_INTERVAL - time_since_yield)

            done, _ = await asyncio.wait([gen_task], timeout=time_to_next_keepalive)

            if done:
                try:
                    event_data = gen_task.result()
                    yield {"data": json.dumps(event_data)}
                    last_yield_time = asyncio.get_event_loop().time()
                    gen_task = None
                except StopAsyncIteration:
                    stream_finished = True
                except Exception as exc:
                    yield {"data": json.dumps({"type": "error", "message": str(exc)})}
                    stream_finished = True
            else:
                # Keepalive timeout expired: send keepalive event
                yield {"data": json.dumps({"type": "keepalive"})}
                last_yield_time = asyncio.get_event_loop().time()

    return EventSourceResponse(event_generator())
