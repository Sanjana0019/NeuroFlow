import asyncio
import json
import logging
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.models.pipeline import PipelineConfig
from pipelines.generation.generator import Generator
from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import Reranker
from pipelines.retrieval.retriever import Retriever

logger = logging.getLogger("neuroflow.api.compare")

router = APIRouter(prefix="/pipelines", tags=["Pipelines & Compare"])


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Common query text sent to both pipelines")
    pipeline_a_id: UUID = Field(..., description="ID of Pipeline A")
    pipeline_b_id: UUID = Field(..., description="ID of Pipeline B")


class PipelineBranchResult(BaseModel):
    run_id: UUID
    pipeline_id: UUID
    pipeline_version: int
    generation: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    chunks_used: int
    evaluation_score: float | None = None


class CompareResponse(BaseModel):
    query: str
    pipeline_a: PipelineBranchResult
    pipeline_b: PipelineBranchResult


async def execute_pipeline_branch(
    pipeline_id: UUID,
    query: str,
    request: Request,
) -> PipelineBranchResult:
    """Execute a single pipeline branch end-to-end with isolated latency measurement and async eval dispatch."""
    db_pool = getattr(request.app.state, "db_pool", None)
    client = getattr(request.app.state, "neuroflow_client", None)
    arq_redis = getattr(request.app.state, "arq_redis", None)

    # 1. Fetch pipeline configuration and version
    version = 1
    top_k = 5
    retrieval_k = 20
    token_budget = 4000
    enable_query_expansion = True
    auto_eval = True

    if db_pool:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, version, status, config FROM pipelines WHERE id = $1",
                pipeline_id,
            )
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pipeline '{pipeline_id}' not found",
                )
            if row["status"] == "archived":
                raise HTTPException(
                    status_code=400,
                    detail=f"Pipeline '{pipeline_id}' is archived",
                )

            version = row["version"] or 1
            cfg_raw = row["config"]
            if isinstance(cfg_raw, str):
                cfg_raw = json.loads(cfg_raw)
            if isinstance(cfg_raw, dict):
                if "retrieval" in cfg_raw:
                    top_k = cfg_raw["retrieval"].get("top_k_after_rerank", top_k)
                    retrieval_k = cfg_raw["retrieval"].get("dense_k", retrieval_k)
                    enable_query_expansion = cfg_raw["retrieval"].get("query_expansion", enable_query_expansion)
                if "generation" in cfg_raw:
                    token_budget = cfg_raw["generation"].get("max_context_tokens", token_budget)
                if "evaluation" in cfg_raw:
                    auto_eval = cfg_raw["evaluation"].get("auto_evaluate", auto_eval)

    # 2. Setup Retrieval Pipeline with configured parameters
    retriever = Retriever(db_pool=db_pool, embedder=client)
    query_processor = QueryProcessor(client=client)
    reranker = Reranker(client=client)
    context_assembler = ContextAssembler(token_budget=token_budget)

    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
        query_processor=query_processor,
        reranker=reranker,
        context_assembler=context_assembler,
        enable_query_expansion=enable_query_expansion,
    )

    # 3. Measure Retrieval Execution
    t_start = time.perf_counter()
    chunks, assembled_context, processed_query = await retrieval_pipeline.run(
        query=query,
        mode="full",
        top_k=top_k,
        retrieval_k=retrieval_k,
        token_budget=token_budget,
    )
    t_retrieval_done = time.perf_counter()
    retrieval_latency_ms = (t_retrieval_done - t_start) * 1000

    # 4. Measure Generation Execution
    generator = Generator(client=client)
    gen_output = await generator.generate(
        query=query,
        assembled_context=assembled_context,
        chunks_used=chunks,
        query_type=processed_query.query_type,
        pipeline_id=pipeline_id,
        pipeline_version=version,
        db_pool=db_pool,
        arq_redis=(arq_redis if auto_eval else None),
    )
    t_gen_done = time.perf_counter()
    generation_latency_ms = (t_gen_done - t_retrieval_done) * 1000
    total_latency_ms = (t_gen_done - t_start) * 1000

    return PipelineBranchResult(
        run_id=gen_output.run_id,
        pipeline_id=pipeline_id,
        pipeline_version=version,
        generation=gen_output.generation,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        chunks_used=len(chunks),
        evaluation_score=None,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_pipelines(
    body: CompareRequest,
    request: Request,
):
    """Execute both pipelines concurrently using asyncio.gather with non-blocking evaluation."""
    query_text = (body.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    branch_a_task = execute_pipeline_branch(body.pipeline_a_id, query_text, request)
    branch_b_task = execute_pipeline_branch(body.pipeline_b_id, query_text, request)

    # Run both branches concurrently in parallel
    result_a, result_b = await asyncio.gather(branch_a_task, branch_b_task)

    return CompareResponse(
        query=query_text,
        pipeline_a=result_a,
        pipeline_b=result_b,
    )
