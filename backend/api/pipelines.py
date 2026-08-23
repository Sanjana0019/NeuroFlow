import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.api.analytics_utils import calculate_percentile, calculate_run_cost
from backend.models.pipeline import PipelineConfig

logger = logging.getLogger("neuroflow.api.pipelines")

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])


class PipelineResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    version: int = 1
    status: str = "active"
    config: dict[str, Any]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_run_metrics: dict[str, Any] | None = None
    evaluation_summary: dict[str, Any] | None = None


class PaginatedRunsResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


class AnalyticsResponse(BaseModel):
    retrieval_latency_p50: float
    retrieval_latency_p95: float
    retrieval_latency_p99: float
    avg_generation_latency_ms: float
    avg_evaluation_scores: dict[str, float | None]
    cost_per_query: float
    queries_per_day: list[dict[str, Any]]


@router.post("", response_model=PipelineResponse, status_code=201)
async def create_pipeline(
    config: PipelineConfig,
    request: Request,
):
    """Create a new named pipeline and initialize version 1."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    pipeline_id = uuid4()
    cfg_dict = config.model_dump()

    async with db_pool.acquire() as conn:
        # Check for unique name conflict
        existing = await conn.fetchval(
            "SELECT id FROM pipelines WHERE name = $1",
            config.name,
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline with name '{config.name}' already exists",
            )

        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO pipelines (
                    id,
                    name,
                    description,
                    version,
                    status,
                    config,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, 1, 'active', $4, NOW(), NOW())
                RETURNING id, name, description, version, status, config, created_at, updated_at
                """,
                pipeline_id,
                config.name,
                config.description,
                json.dumps(cfg_dict),
            )

            # Insert initial version 1 record
            await conn.execute(
                """
                INSERT INTO pipeline_versions (
                    pipeline_id,
                    version,
                    config,
                    created_at
                )
                VALUES ($1, $2, $3, NOW())
                """,
                pipeline_id,
                1,
                json.dumps(cfg_dict),
            )

    return PipelineResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        version=row["version"],
        status=row["status"],
        config=json.loads(row["config"]) if isinstance(row["config"], str) else row["config"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("", response_model=list[PipelineResponse])
async def list_pipelines(
    request: Request,
    include_archived: bool = Query(default=False),
):
    """List pipelines with last-run metrics."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    status_filter = "" if include_archived else "WHERE p.status != 'archived'"

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT p.id, p.name, p.description, p.version, p.status, p.config, p.created_at, p.updated_at,
                   COUNT(r.id) AS total_runs,
                   MAX(r.created_at) AS last_run_at,
                   (SELECT status FROM pipeline_runs WHERE pipeline_id = p.id ORDER BY created_at DESC LIMIT 1) AS last_run_status,
                   (SELECT latency_ms FROM pipeline_runs WHERE pipeline_id = p.id ORDER BY created_at DESC LIMIT 1) AS last_run_latency_ms
            FROM pipelines p
            LEFT JOIN pipeline_runs r ON r.pipeline_id = p.id
            {status_filter}
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        )

    results = []
    for r in rows:
        cfg = json.loads(r["config"]) if isinstance(r["config"], str) else r["config"]
        metrics = {
            "total_runs": r["total_runs"] or 0,
            "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
            "last_run_status": r["last_run_status"],
            "last_run_latency_ms": r["last_run_latency_ms"],
        }
        results.append(
            PipelineResponse(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                version=r["version"],
                status=r["status"],
                config=cfg,
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                last_run_metrics=metrics,
            )
        )
    return results


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: UUID,
    request: Request,
):
    """Get full current configuration and aggregate evaluation information for a pipeline."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, description, version, status, config, created_at, updated_at
            FROM pipelines
            WHERE id = $1
            """,
            pipeline_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

        # Fetch aggregate evaluation metrics
        eval_row = await conn.fetchrow(
            """
            SELECT
                COUNT(r.id) AS total_runs,
                AVG(e.faithfulness) AS avg_faithfulness,
                AVG(e.answer_relevance) AS avg_answer_relevance,
                AVG(e.context_precision) AS avg_context_precision,
                AVG(e.context_recall) AS avg_context_recall,
                AVG(e.overall_score) AS avg_overall_score
            FROM pipeline_runs r
            LEFT JOIN evaluations e ON e.run_id = r.id
            WHERE r.pipeline_id = $1
            """,
            pipeline_id,
        )

    cfg = json.loads(row["config"]) if isinstance(row["config"], str) else row["config"]
    eval_summary = {
        "total_runs": eval_row["total_runs"] or 0,
        "avg_faithfulness": round(float(eval_row["avg_faithfulness"]), 4) if eval_row["avg_faithfulness"] is not None else None,
        "avg_answer_relevance": round(float(eval_row["avg_answer_relevance"]), 4) if eval_row["avg_answer_relevance"] is not None else None,
        "avg_context_precision": round(float(eval_row["avg_context_precision"]), 4) if eval_row["avg_context_precision"] is not None else None,
        "avg_context_recall": round(float(eval_row["avg_context_recall"]), 4) if eval_row["avg_context_recall"] is not None else None,
        "avg_overall_score": round(float(eval_row["avg_overall_score"]), 4) if eval_row["avg_overall_score"] is not None else None,
    }

    return PipelineResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        version=row["version"],
        status=row["status"],
        config=cfg,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        evaluation_summary=eval_summary,
    )


@router.patch("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: UUID,
    config: PipelineConfig,
    request: Request,
):
    """Update pipeline configuration, creating a new immutable version and preserving history."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    cfg_dict = config.model_dump()

    async with db_pool.acquire() as conn:
        current = await conn.fetchrow(
            "SELECT id, version, status FROM pipelines WHERE id = $1",
            pipeline_id,
        )
        if not current:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
        if current["status"] == "archived":
            raise HTTPException(status_code=400, detail="Cannot update an archived pipeline")

        new_version = (current["version"] or 1) + 1

        async with conn.transaction():
            # Update pipelines table with incremented version and new config
            row = await conn.fetchrow(
                """
                UPDATE pipelines
                SET name = $1,
                    description = $2,
                    version = $3,
                    config = $4,
                    updated_at = NOW()
                WHERE id = $5
                RETURNING id, name, description, version, status, config, created_at, updated_at
                """,
                config.name,
                config.description,
                new_version,
                json.dumps(cfg_dict),
                pipeline_id,
            )

            # Insert new immutable version record
            await conn.execute(
                """
                INSERT INTO pipeline_versions (
                    pipeline_id,
                    version,
                    config,
                    created_at
                )
                VALUES ($1, $2, $3, NOW())
                """,
                pipeline_id,
                new_version,
                json.dumps(cfg_dict),
            )

    return PipelineResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        version=row["version"],
        status=row["status"],
        config=json.loads(row["config"]) if isinstance(row["config"], str) else row["config"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.delete("/{pipeline_id}")
async def delete_pipeline(
    pipeline_id: UUID,
    request: Request,
):
    """Soft delete pipeline by setting status='archived' without physically removing data."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE pipelines
            SET status = 'archived',
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, status
            """,
            pipeline_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    return {
        "message": "Pipeline archived",
        "id": str(pipeline_id),
        "status": "archived",
    }


@router.get("/{pipeline_id}/runs", response_model=PaginatedRunsResponse)
async def get_pipeline_runs(
    pipeline_id: UUID,
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Paginated pipeline run history."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    offset = (page - 1) * page_size

    async with db_pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM pipeline_runs WHERE pipeline_id = $1",
            pipeline_id,
        ) or 0

        rows = await conn.fetch(
            """
            SELECT r.id AS run_id, r.pipeline_id, r.pipeline_version, r.query, r.generation,
                   r.latency_ms, r.input_tokens, r.output_tokens, r.model_used, r.status, r.created_at,
                   e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score
            FROM pipeline_runs r
            LEFT JOIN evaluations e ON e.run_id = r.id
            WHERE r.pipeline_id = $1
            ORDER BY r.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            pipeline_id,
            page_size,
            offset,
        )

    items = []
    for r in rows:
        eval_dict = None
        if r["overall_score"] is not None or r["faithfulness"] is not None:
            eval_dict = {
                "faithfulness": r["faithfulness"],
                "answer_relevance": r["answer_relevance"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
                "overall_score": r["overall_score"],
            }
        items.append(
            {
                "run_id": str(r["run_id"]),
                "pipeline_id": str(r["pipeline_id"]),
                "pipeline_version": r["pipeline_version"] or 1,
                "query": r["query"],
                "generation": r["generation"],
                "latency_ms": r["latency_ms"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "model_used": r["model_used"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "evaluation": eval_dict,
            }
        )

    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

    return PaginatedRunsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{pipeline_id}/analytics", response_model=AnalyticsResponse)
async def get_pipeline_analytics(
    pipeline_id: UUID,
    request: Request,
):
    """Retrieve statistical analytics: p50/p95/p99 latency, cost/query, averages, and 30-day activity."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    async with db_pool.acquire() as conn:
        # Fetch runs with latency and tokens
        runs = await conn.fetch(
            """
            SELECT r.id, r.latency_ms, r.input_tokens, r.output_tokens, r.model_used, r.created_at,
                   e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, e.overall_score
            FROM pipeline_runs r
            LEFT JOIN evaluations e ON e.run_id = r.id
            WHERE r.pipeline_id = $1
            ORDER BY r.created_at DESC
            """,
            pipeline_id,
        )

        # Queries per day for last 30 days
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        daily_rows = await conn.fetch(
            """
            SELECT DATE(created_at) AS run_date, COUNT(*) AS count
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND created_at >= $2
            GROUP BY DATE(created_at)
            ORDER BY run_date ASC
            """,
            pipeline_id,
            thirty_days_ago,
        )

    latencies = [float(r["latency_ms"]) for r in runs if r["latency_ms"] is not None]
    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)
    p99 = calculate_percentile(latencies, 99.0)
    avg_gen_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Calculate average evaluation metrics
    f_scores = [r["faithfulness"] for r in runs if r["faithfulness"] is not None]
    r_scores = [r["answer_relevance"] for r in runs if r["answer_relevance"] is not None]
    cp_scores = [r["context_precision"] for r in runs if r["context_precision"] is not None]
    cr_scores = [r["context_recall"] for r in runs if r["context_recall"] is not None]
    ov_scores = [r["overall_score"] for r in runs if r["overall_score"] is not None]

    avg_eval = {
        "faithfulness": round(sum(f_scores) / len(f_scores), 4) if f_scores else None,
        "answer_relevance": round(sum(r_scores) / len(r_scores), 4) if r_scores else None,
        "context_precision": round(sum(cp_scores) / len(cp_scores), 4) if cp_scores else None,
        "context_recall": round(sum(cr_scores) / len(cr_scores), 4) if cr_scores else None,
        "overall_score": round(sum(ov_scores) / len(ov_scores), 4) if ov_scores else None,
    }

    # Cost per query
    total_cost = sum(
        calculate_run_cost(
            input_tokens=r["input_tokens"] or 0,
            output_tokens=r["output_tokens"] or 0,
            model_used=r["model_used"],
        )
        for r in runs
    )
    cost_per_query = round(total_cost / len(runs), 6) if runs else 0.0

    queries_per_day = [
        {"date": str(row["run_date"]), "count": row["count"]}
        for row in daily_rows
    ]

    return AnalyticsResponse(
        retrieval_latency_p50=p50,
        retrieval_latency_p95=p95,
        retrieval_latency_p99=p99,
        avg_generation_latency_ms=avg_gen_latency,
        avg_evaluation_scores=avg_eval,
        cost_per_query=cost_per_query,
        queries_per_day=queries_per_day,
    )
