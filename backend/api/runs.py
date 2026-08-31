import asyncio
from datetime import datetime
import json
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("neuroflow.api.runs")

router = APIRouter(tags=["Runs & Evaluations"])


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Human feedback rating from 1 to 5")


class EvaluationResponse(BaseModel):
    run_id: UUID
    query: str | None = None
    generation: str | None = None
    pipeline_name: str | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    overall_score: float | None = None
    judge_model: str | None = None
    user_rating: int | None = None
    calibration_needed: bool = False
    evaluated_at: datetime | None = None
    chunks: list[dict[str, Any]] = []


@router.get("/evaluations/stream")
async def stream_evaluations(request: Request):
    """Server-Sent Events (SSE) stream forwarding real-time evaluations from Redis pub/sub channel."""
    redis_client = getattr(request.app.state, "redis", None) or getattr(request.app.state, "arq_redis", None)

    async def event_generator():
        if redis_client is not None:
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("evaluations:new")
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        yield {"data": data}
                    else:
                        # Send keepalive comment/ping
                        await asyncio.sleep(0.5)
            finally:
                await pubsub.unsubscribe("evaluations:new")
                await pubsub.aclose()
        else:
            # Fallback dummy stream when Redis is not active
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(10.0)
                yield {"data": json.dumps({"type": "keepalive"})}

    return EventSourceResponse(event_generator())


@router.get("/evaluations", response_model=list[EvaluationResponse])
async def list_evaluations(
    request: Request,
    pipeline_id: UUID | None = None,
    min_overall: float | None = Query(None, ge=0.0, le=1.0),
    min_faithfulness: float | None = Query(None, ge=0.0, le=1.0),
    min_relevance: float | None = Query(None, ge=0.0, le=1.0),
    min_precision: float | None = Query(None, ge=0.0, le=1.0),
    min_recall: float | None = Query(None, ge=0.0, le=1.0),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Retrieve historical evaluations with optional pipeline, score thresholds, and date filters."""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable")

    query_parts = [
        """
        SELECT e.id, e.run_id, e.faithfulness, e.answer_relevance, e.context_precision,
               e.context_recall, e.overall_score, e.judge_model, e.user_rating, e.evaluated_at,
               r.query, r.generation, r.retrieved_chunk_ids, p.name AS pipeline_name
        FROM evaluations e
        JOIN pipeline_runs r ON r.id = e.run_id
        LEFT JOIN pipelines p ON p.id = r.pipeline_id
        WHERE 1=1
        """
    ]
    params: list[Any] = []
    idx = 1

    if pipeline_id:
        query_parts.append(f"AND r.pipeline_id = ${idx}")
        params.append(pipeline_id)
        idx += 1

    if min_overall is not None:
        query_parts.append(f"AND e.overall_score >= ${idx}")
        params.append(min_overall)
        idx += 1

    if min_faithfulness is not None:
        query_parts.append(f"AND e.faithfulness >= ${idx}")
        params.append(min_faithfulness)
        idx += 1

    if min_relevance is not None:
        query_parts.append(f"AND e.answer_relevance >= ${idx}")
        params.append(min_relevance)
        idx += 1

    if min_precision is not None:
        query_parts.append(f"AND e.context_precision >= ${idx}")
        params.append(min_precision)
        idx += 1

    if min_recall is not None:
        query_parts.append(f"AND e.context_recall >= ${idx}")
        params.append(min_recall)
        idx += 1

    if start_date:
        query_parts.append(f"AND e.evaluated_at >= ${idx}")
        params.append(start_date)
        idx += 1

    if end_date:
        query_parts.append(f"AND e.evaluated_at <= ${idx}")
        params.append(end_date)
        idx += 1

    query_parts.append(f"ORDER BY e.evaluated_at DESC LIMIT ${idx}")
    params.append(limit)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(" ".join(query_parts), *params)

        results: list[EvaluationResponse] = []
        for r in rows:
            chunks: list[dict[str, Any]] = []
            if r["retrieved_chunk_ids"]:
                chunk_rows = await conn.fetch(
                    "SELECT id, content, chunk_index FROM chunks WHERE id = ANY($1::uuid[])",
                    r["retrieved_chunk_ids"],
                )
                chunks = [{"id": str(c["id"]), "content": c["content"], "chunk_index": c["chunk_index"]} for c in chunk_rows]

            results.append(
                EvaluationResponse(
                    run_id=r["run_id"],
                    query=r["query"],
                    generation=r["generation"],
                    pipeline_name=r["pipeline_name"] or "Default Pipeline",
                    faithfulness=r["faithfulness"],
                    answer_relevance=r["answer_relevance"],
                    context_precision=r["context_precision"],
                    context_recall=r["context_recall"],
                    overall_score=r["overall_score"],
                    judge_model=r["judge_model"],
                    user_rating=r["user_rating"],
                    evaluated_at=r["evaluated_at"],
                    chunks=chunks,
                )
            )

        return results


@router.get("/runs/{run_id}/evaluation", response_model=EvaluationResponse)
async def get_run_evaluation(
    run_id: UUID,
    request: Request,
):
    """Retrieve evaluation scores and retrieved chunks for a specific pipeline run."""
    db_pool = request.app.state.db_pool
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT e.id, e.run_id, e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall,
                   e.overall_score, e.judge_model, e.user_rating, e.evaluated_at,
                   r.query, r.generation, r.retrieved_chunk_ids, p.name AS pipeline_name
            FROM evaluations e
            JOIN pipeline_runs r ON r.id = e.run_id
            LEFT JOIN pipelines p ON p.id = r.pipeline_id
            WHERE e.run_id = $1
            ORDER BY e.evaluated_at DESC
            LIMIT 1
            """,
            run_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail=f"Evaluation for run '{run_id}' not found yet")

        chunks: list[dict[str, Any]] = []
        if row["retrieved_chunk_ids"]:
            chunk_rows = await conn.fetch(
                "SELECT id, content, chunk_index FROM chunks WHERE id = ANY($1::uuid[])",
                row["retrieved_chunk_ids"],
            )
            chunks = [{"id": str(c["id"]), "content": c["content"], "chunk_index": c["chunk_index"]} for c in chunk_rows]

        return EvaluationResponse(
            run_id=row["run_id"],
            query=row["query"],
            generation=row["generation"],
            pipeline_name=row["pipeline_name"] or "Default Pipeline",
            faithfulness=row["faithfulness"],
            answer_relevance=row["answer_relevance"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
            overall_score=row["overall_score"],
            judge_model=row["judge_model"],
            user_rating=row["user_rating"],
            evaluated_at=row["evaluated_at"],
            chunks=chunks,
        )


@router.patch("/runs/{run_id}/rating", response_model=EvaluationResponse)
async def update_run_human_rating(
    run_id: UUID,
    body: RatingRequest,
    request: Request,
):
    """Update human rating for a run and detect calibration discrepancies."""
    if body.rating < 1 or body.rating > 5:
        raise HTTPException(
            status_code=400,
            detail="Rating must be an integer between 1 and 5",
        )

    db_pool = request.app.state.db_pool

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, run_id, faithfulness, answer_relevance, context_precision, context_recall,
                   overall_score, judge_model, user_rating
            FROM evaluations
            WHERE run_id = $1
            ORDER BY evaluated_at DESC
            LIMIT 1
            """,
            run_id,
        )

        if not row:
            run_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pipeline_runs WHERE id = $1)",
                run_id,
            )
            if not run_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pipeline run '{run_id}' not found",
                )

            await conn.execute(
                """
                INSERT INTO evaluations (run_id, user_rating)
                VALUES ($1, $2)
                """,
                run_id,
                body.rating,
            )
            return EvaluationResponse(
                run_id=run_id,
                user_rating=body.rating,
                calibration_needed=False,
            )

        await conn.execute(
            """
            UPDATE evaluations
            SET user_rating = $1
            WHERE id = $2
            """,
            body.rating,
            row["id"],
        )

        overall_score = row["overall_score"]
        calibration_needed = False

        if overall_score is not None:
            human_score = body.rating / 5.0
            discrepancy = abs(overall_score - human_score)
            if discrepancy > 0.3:
                calibration_needed = True

        return EvaluationResponse(
            run_id=run_id,
            faithfulness=row["faithfulness"],
            answer_relevance=row["answer_relevance"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
            overall_score=overall_score,
            judge_model=row["judge_model"],
            user_rating=body.rating,
            calibration_needed=calibration_needed,
        )
