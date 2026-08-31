import asyncio
from datetime import datetime
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

try:
    from backend.config import settings
except ImportError:
    from config import settings

from pipelines.finetuning.extractor import TrainingDataExtractor
from pipelines.finetuning.job_manager import FineTuningJobManager
from pipelines.finetuning.tracker import FineTuningTracker

logger = logging.getLogger("neuroflow.api.finetune")

router = APIRouter(prefix="/finetune", tags=["Fine-Tuning"])


class CreateJobRequest(BaseModel):
    base_model: str = Field(default="gpt-4o-mini-2024-07-18", description="Base model to fine-tune")
    min_quality_score: float = Field(default=0.82, ge=0.0, le=1.0, description="Minimum quality score threshold")


class JobResponse(BaseModel):
    job_id: UUID
    provider_job_id: str | None = None
    base_model: str
    fine_tuned_model: str | None = None
    status: str
    training_pair_count: int
    mlflow_run_id: str | None = None
    metrics: dict[str, Any] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class PreviewPairItem(BaseModel):
    pair_id: str
    run_id: str
    quality_score: float | None = None
    faithfulness: float | None = None
    created_at: str | None = None
    messages: list[dict[str, str]]


class TrainingPairDetailItem(BaseModel):
    id: str
    run_id: str
    user_message: str
    assistant_message: str
    system_prompt: str
    quality_score: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    token_count: int
    has_citation: bool
    has_pii: bool
    user_rating: int | None = None
    is_valid: bool
    rejection_reason: str | None = None
    included_in_job: str | None = None
    created_at: str | None = None


class ValidationRuleItem(BaseModel):
    name: str
    requirement: str
    passed: bool


class DatasetReadinessResponse(BaseModel):
    total_candidates: int
    eligible_sft_count: int
    min_required_for_finetuning: int
    remaining_for_finetuning: int
    can_export: bool
    can_finetune: bool
    openai_configured: bool
    dpo_pair_count: int
    validation_rules: list[ValidationRuleItem]
    rejected_count: int
    rejection_summary: dict[str, int]


class DPOPreviewItem(BaseModel):
    prompt: str
    chosen: str
    rejected: str


@router.get("/readiness", response_model=DatasetReadinessResponse)
async def get_dataset_readiness(request: Request):
    """Retrieve canonical dataset readiness metrics, validation checklist, and eligibility counts."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    extractor = TrainingDataExtractor()
    openai_configured = bool(settings.openai_api_key and settings.openai_api_key.strip())
    readiness = await extractor.get_dataset_readiness(
        db_pool=db_pool,
        openai_configured=openai_configured,
    )
    return readiness


@router.get("/training-data", response_model=list[TrainingPairDetailItem])
async def list_training_data(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Retrieve all collected training pairs with real token counts, citations, and validation metadata."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    extractor = TrainingDataExtractor()
    pairs = await extractor.get_all_training_pairs(db_pool=db_pool, limit=limit)
    return pairs


@router.get("/training-data/preview", response_model=list[PreviewPairItem])
async def preview_training_data(request: Request):
    """Preview up to 5 currently eligible training pairs without creating or submitting a job."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    extractor = TrainingDataExtractor()
    pairs = await extractor.preview_eligible_pairs(db_pool=db_pool, limit=5)
    return pairs


@router.get("/dpo/preview", response_model=list[DPOPreviewItem])
async def preview_dpo_data(request: Request):
    """Preview up to 5 currently eligible DPO preference pairs."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    extractor = TrainingDataExtractor()
    pairs = await extractor.preview_dpo_pairs(db_pool=db_pool, limit=5)
    return pairs


@router.get("/datasets/export")
async def export_dataset_endpoint(
    request: Request,
    dataset_type: str = Query(default="sft", pattern="^(sft|dpo)$"),
    format: str = Query(default="jsonl", pattern="^(jsonl|json)$"),
):
    """Export and download validated SFT or DPO datasets in JSONL or JSON format."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    extractor = TrainingDataExtractor()
    content, media_type, count = await extractor.export_dataset(
        db_pool=db_pool,
        dataset_type=dataset_type,
        export_format=format,
    )

    filename = f"neuroflow_{dataset_type}_dataset.{format}"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Total-Records": str(count),
        },
    )


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_finetune_job(
    body: CreateJobRequest,
    request: Request,
):
    """Extract eligible training pairs, log MLflow run, and submit OpenAI fine-tuning job."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    # Validate provider configuration
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is not configured for fine-tuning. Please configure OPENAI_API_KEY.",
        )

    client = getattr(request.app.state, "neuroflow_client", None)
    arq_redis = getattr(request.app.state, "arq_redis", None)
    job_id = uuid4()
    base_model = body.base_model or "gpt-4o-mini-2024-07-18"

    # 1. Extract & validate training data
    extractor = TrainingDataExtractor()
    valid_pairs, rejected_pairs, jsonl_path = await extractor.extract_and_validate(
        db_pool=db_pool,
        job_id=job_id,
        min_quality_score=body.min_quality_score,
        client=client,
    )

    if len(valid_pairs) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient training pairs. Found {len(valid_pairs)} eligible pairs (minimum 10 required)",
        )

    # Calculate quality metrics & date range
    scores = [p["quality_score"] for p in valid_pairs if p.get("quality_score") is not None]
    avg_quality = sum(scores) / len(scores) if scores else 0.85
    dates = [p["created_at"] for p in valid_pairs if p.get("created_at")]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "all-time"

    # 2. Insert initial job record into database
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO finetune_jobs (
                id,
                base_model,
                status,
                training_pair_count
            )
            VALUES ($1, $2, 'pending', $3)
            """,
            job_id,
            base_model,
            len(valid_pairs),
        )

    # 3. Create MLflow tracking run
    tracker = FineTuningTracker()
    mlflow_run_id = await tracker.create_job_run(
        job_id=str(job_id),
        base_model=base_model,
        training_pair_count=len(valid_pairs),
        avg_quality_score=avg_quality,
        date_range=date_range,
        jsonl_path=jsonl_path,
    )

    # 4. Submit fine-tuning job to OpenAI
    job_manager = FineTuningJobManager(tracker=tracker)
    try:
        provider_job_id = await job_manager.submit_job(
            job_id=job_id,
            jsonl_path=jsonl_path,
            base_model=base_model,
        )
    except Exception as exc:
        logger.error("OpenAI fine-tuning submission failed for job %s: %s", job_id, exc)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE finetune_jobs SET status = 'failed' WHERE id = $1",
                job_id,
            )
        raise HTTPException(status_code=502, detail=f"OpenAI submission failed: {exc}")

    # 5. Update database record with provider job ID and mlflow_run_id
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE finetune_jobs
            SET provider_job_id = $1,
                mlflow_run_id = $2,
                status = 'running'
            WHERE id = $3
            RETURNING id, provider_job_id, base_model, fine_tuned_model, status,
                      training_pair_count, mlflow_run_id, metrics, created_at, completed_at
            """,
            provider_job_id,
            mlflow_run_id,
            job_id,
        )

    # 6. Dispatch asynchronous background polling without blocking HTTP response
    if arq_redis:
        async def _dispatch_poll():
            try:
                await arq_redis.enqueue_job("poll_finetune_job", job_id=str(job_id), _defer_by=60)
            except Exception as exc:
                logger.warning("Failed to enqueue initial poll_finetune_job: %s", exc)

        asyncio.create_task(_dispatch_poll())

    return JobResponse(
        job_id=row["id"],
        provider_job_id=row["provider_job_id"],
        base_model=row["base_model"],
        fine_tuned_model=row["fine_tuned_model"],
        status=row["status"],
        training_pair_count=row["training_pair_count"],
        mlflow_run_id=row["mlflow_run_id"],
        metrics=json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_finetune_jobs(request: Request):
    """List all historical and active fine-tuning jobs."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, provider_job_id, base_model, fine_tuned_model, status,
                   training_pair_count, mlflow_run_id, metrics, created_at, completed_at
            FROM finetune_jobs
            ORDER BY created_at DESC
            """
        )

    return [
        JobResponse(
            job_id=r["id"],
            provider_job_id=r["provider_job_id"],
            base_model=r["base_model"],
            fine_tuned_model=r["fine_tuned_model"],
            status=r["status"],
            training_pair_count=r["training_pair_count"] or 0,
            mlflow_run_id=r["mlflow_run_id"],
            metrics=json.loads(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"],
            created_at=r["created_at"],
            completed_at=r["completed_at"],
        )
        for r in rows
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_finetune_job(job_id: UUID, request: Request):
    """Retrieve fine-tuning job details by ID."""
    db_pool = getattr(request.app.state, "db_pool", None)
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection pool is not available")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, provider_job_id, base_model, fine_tuned_model, status,
                   training_pair_count, mlflow_run_id, metrics, created_at, completed_at
            FROM finetune_jobs
            WHERE id = $1
            """,
            job_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found")

    return JobResponse(
        job_id=row["id"],
        provider_job_id=row["provider_job_id"],
        base_model=row["base_model"],
        fine_tuned_model=row["fine_tuned_model"],
        status=row["status"],
        training_pair_count=row["training_pair_count"] or 0,
        mlflow_run_id=row["mlflow_run_id"],
        metrics=json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )
