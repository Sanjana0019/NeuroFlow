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


class PipelineSummaryMetrics(BaseModel):
    has_data: bool = False
    quality_score: float | None = None
    quality_label: str = "Overall Quality"
    faithfulness: float | None = None
    faithfulness_change_pct: float | None = None
    queries_7d: int = 0
    queries_previous_7d: int = 0
    queries_change_pct: float | None = None
    latency_p50_ms: float | None = None
    latency_change_pct: float | None = None
    trend_7d: list[dict[str, Any]] = Field(default_factory=list)


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
    metrics_summary: PipelineSummaryMetrics | None = None



class PaginatedRunsResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


class AnalyticsResponse(BaseModel):
    pipeline_id: UUID
    pipeline_name: str
    total_runs: int
    retrieval_latency_p50: float
    retrieval_latency_p95: float
    retrieval_latency_p99: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_generation_latency_ms: float
    avg_latency_ms: float
    avg_evaluation_scores: dict[str, float | None]
    cost_per_query: float
    cost_per_query_usd: float
    total_cost_usd: float
    queries_per_day: list[dict[str, Any]]
    recent_failures: list[dict[str, Any]]


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


async def compute_pipeline_summary_metrics(conn, pipeline_id: UUID, now: datetime) -> PipelineSummaryMetrics:
    """Compute real 7-day, previous 7-day, and trend metrics for a specific pipeline from PostgreSQL."""
    seven_days_ago = now - timedelta(days=7)
    fourteen_days_ago = now - timedelta(days=14)

    default_trend = []
    for i in range(6, -1, -1):
        d_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        default_trend.append({"date": d_str, "query_count": 0})

    try:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(CASE WHEN r.created_at >= $2 THEN 1 END) AS count_7d,
                COUNT(CASE WHEN r.created_at >= $3 AND r.created_at < $2 THEN 1 END) AS count_prev_7d,
                AVG(CASE WHEN r.created_at >= $2 THEN e.overall_score END) AS quality_7d,
                AVG(e.overall_score) AS quality_all_time,
                AVG(CASE WHEN r.created_at >= $2 THEN e.faithfulness END) AS faith_7d,
                AVG(e.faithfulness) AS faith_all_time,
                AVG(CASE WHEN r.created_at >= $3 AND r.created_at < $2 THEN e.faithfulness END) AS faith_prev_7d,
                COUNT(r.id) AS total_runs
            FROM pipeline_runs r
            LEFT JOIN evaluations e ON e.run_id = r.id
            WHERE r.pipeline_id = $1
            """,
            pipeline_id, seven_days_ago, fourteen_days_ago
        )
    except Exception as exc:
        logger.warning("Error fetching 7d summary row for pipeline %s: %s", pipeline_id, exc)
        row = None

    if not row or (row.get("total_runs") or 0) == 0:
        return PipelineSummaryMetrics(
            has_data=False,
            quality_score=None,
            quality_label="Overall Quality",
            faithfulness=None,
            faithfulness_change_pct=None,
            queries_7d=0,
            queries_previous_7d=0,
            queries_change_pct=None,
            latency_p50_ms=None,
            latency_change_pct=None,
            trend_7d=default_trend,
        )

    count_7d = row["count_7d"] or 0
    count_prev_7d = row["count_prev_7d"] or 0
    query_change_pct = None
    if count_prev_7d > 0:
        query_change_pct = round(((count_7d - count_prev_7d) / count_prev_7d) * 100, 1)

    lat_7d = None
    lat_prev_7d = None
    try:
        lat_7d = await conn.fetchval(
            """
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY r.latency_ms)
            FROM pipeline_runs r
            WHERE r.pipeline_id = $1 AND r.created_at >= $2
            """,
            pipeline_id, seven_days_ago
        )
        if lat_7d is None and (row["total_runs"] or 0) > 0:
            lat_7d = await conn.fetchval(
                """
                SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY r.latency_ms)
                FROM pipeline_runs r
                WHERE r.pipeline_id = $1
                """,
                pipeline_id
            )

        lat_prev_7d = await conn.fetchval(
            """
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY r.latency_ms)
            FROM pipeline_runs r
            WHERE r.pipeline_id = $1 AND r.created_at >= $3 AND r.created_at < $2
            """,
            pipeline_id, seven_days_ago, fourteen_days_ago
        )
    except Exception as exc:
        logger.debug("Could not calculate percentile latency for pipeline %s: %s", pipeline_id, exc)

    lat_change_pct = None
    if lat_prev_7d and lat_7d and lat_prev_7d > 0:
        lat_change_pct = round(((lat_7d - lat_prev_7d) / lat_prev_7d) * 100, 1)

    quality_score = row["quality_7d"] if row["quality_7d"] is not None else row.get("quality_all_time")
    faith_7d = row["faith_7d"] if row["faith_7d"] is not None else row.get("faith_all_time")
    faith_prev_7d = row.get("faith_prev_7d")
    faith_change_pct = None
    if faith_prev_7d and faith_7d and faith_prev_7d > 0:
        faith_change_pct = round(((faith_7d - faith_prev_7d) / faith_prev_7d) * 100, 1)

    # 7-day daily trend
    real_trend_7d = default_trend
    try:
        daily_rows = await conn.fetch(
            """
            SELECT DATE(created_at AT TIME ZONE 'UTC') AS day, COUNT(*) AS count
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND created_at >= $2
            GROUP BY DATE(created_at AT TIME ZONE 'UTC')
            ORDER BY day ASC
            """,
            pipeline_id, seven_days_ago
        )
        if daily_rows:
            daily_map = {str(r["day"]): r["count"] for r in daily_rows}
            real_trend_7d = []
            for i in range(6, -1, -1):
                d_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                real_trend_7d.append({"date": d_str, "query_count": daily_map.get(d_str, 0)})
    except Exception as exc:
        logger.debug("Could not fetch daily trend for pipeline %s: %s", pipeline_id, exc)

    return PipelineSummaryMetrics(
        has_data=(row["total_runs"] or 0) > 0,
        quality_score=round(quality_score, 4) if quality_score is not None else None,
        quality_label="Overall Quality",
        faithfulness=round(faith_7d, 4) if faith_7d is not None else None,
        faithfulness_change_pct=faith_change_pct,
        queries_7d=count_7d,
        queries_previous_7d=count_prev_7d,
        queries_change_pct=query_change_pct,
        latency_p50_ms=round(lat_7d, 1) if lat_7d is not None else None,
        latency_change_pct=lat_change_pct,
        trend_7d=real_trend_7d,
    )


@router.get("", response_model=list[PipelineResponse])
async def list_pipelines(
    request: Request,
    include_archived: bool = Query(default=False),
):
    """List pipelines with last-run metrics and real 7-day performance metrics."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not available")

    status_filter = "" if include_archived else "WHERE p.status != 'archived'"
    now = datetime.now(timezone.utc)

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
            last_run = {
                "total_runs": r["total_runs"] or 0,
                "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
                "last_run_status": r["last_run_status"],
                "last_run_latency_ms": r["last_run_latency_ms"],
            }
            summary = await compute_pipeline_summary_metrics(conn, r["id"], now)
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
                    last_run_metrics=last_run,
                    metrics_summary=summary,
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

    now = datetime.now(timezone.utc)

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

        summary = await compute_pipeline_summary_metrics(conn, pipeline_id, now)

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

        eval_summary = None
        if eval_row and eval_row["total_runs"] and eval_row["total_runs"] > 0:
            eval_summary = {
                "total_runs": eval_row["total_runs"],
                "avg_faithfulness": eval_row["avg_faithfulness"],
                "avg_answer_relevance": eval_row["avg_answer_relevance"],
                "avg_context_precision": eval_row["avg_context_precision"],
                "avg_context_recall": eval_row["avg_context_recall"],
                "avg_overall_score": eval_row["avg_overall_score"],
            }

        return PipelineResponse(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            version=row["version"],
            status=row["status"],
            config=json.loads(row["config"]) if isinstance(row["config"], str) else row["config"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            evaluation_summary=eval_summary,
            metrics_summary=summary,
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
        # Fetch pipeline metadata
        pipeline_row = await conn.fetchrow("SELECT name FROM pipelines WHERE id = $1", pipeline_id)
        pipeline_name = pipeline_row["name"] if pipeline_row else "Unknown Pipeline"

        # Fetch runs with latency, tokens, and evals
        runs = await conn.fetch(
            """
            SELECT r.id, r.latency_ms, r.input_tokens, r.output_tokens, r.model_used, r.status, r.created_at,
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

        # Recent failures (status='failed')
        failed_rows = await conn.fetch(
            """
            SELECT id AS run_id, query, created_at AS timestamp, COALESCE(generation, 'Execution failed') AS error_message
            FROM pipeline_runs
            WHERE pipeline_id = $1 AND status = 'failed'
            ORDER BY created_at DESC
            LIMIT 5
            """,
            pipeline_id,
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

    # Cost per query & total cost
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

    recent_failures = [
        {
            "run_id": str(row["run_id"]),
            "query": row["query"],
            "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            "error_message": row["error_message"],
        }
        for row in failed_rows
    ]

    return AnalyticsResponse(
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        total_runs=len(runs),
        retrieval_latency_p50=p50,
        retrieval_latency_p95=p95,
        retrieval_latency_p99=p99,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        avg_generation_latency_ms=avg_gen_latency,
        avg_latency_ms=avg_gen_latency,
        avg_evaluation_scores=avg_eval,
        cost_per_query=cost_per_query,
        cost_per_query_usd=cost_per_query,
        total_cost_usd=round(total_cost, 6),
        queries_per_day=queries_per_day,
        recent_failures=recent_failures,
    )
