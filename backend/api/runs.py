import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("neuroflow.api.runs")

router = APIRouter(tags=["Runs & Evaluations"])


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Human feedback rating from 1 to 5")


class EvaluationResponse(BaseModel):
    run_id: UUID
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    overall_score: float | None = None
    judge_model: str | None = None
    user_rating: int | None = None
    calibration_needed: bool = False


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
        # Check if evaluation exists
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
            # Check if pipeline run exists
            run_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pipeline_runs WHERE id = $1)",
                run_id,
            )
            if not run_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pipeline run '{run_id}' not found",
                )

            # Insert evaluation row with human rating
            eval_id = await conn.fetchval(
                """
                INSERT INTO evaluations (run_id, user_rating)
                VALUES ($1, $2)
                RETURNING id
                """,
                run_id,
                body.rating,
            )
            return EvaluationResponse(
                run_id=run_id,
                user_rating=body.rating,
                calibration_needed=False,
            )

        # Update existing evaluation with user rating
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
