import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI

try:
    from backend.config import settings
    from backend.providers.router import ModelRouter
except ImportError:
    from config import settings
    from providers.router import ModelRouter

from pipelines.finetuning.tracker import FineTuningTracker

logger = logging.getLogger("neuroflow.finetuning.job_manager")


class FineTuningJobManager:
    """Manages OpenAI fine-tuning job submissions, background polling, and ModelRouter registration."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        tracker: FineTuningTracker | None = None,
    ):
        self.api_key = openai_api_key or settings.openai_api_key
        if openai_client:
            self.client = openai_client
        elif self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.tracker = tracker or FineTuningTracker()

    async def submit_job(
        self,
        job_id: UUID | str,
        jsonl_path: str | Path,
        base_model: str = "gpt-4o-mini-2024-07-18",
    ) -> str:
        """Upload training dataset file to OpenAI and submit fine-tuning job."""
        if not self.client:
            raise ValueError("OpenAI API key is not configured for fine-tuning")

        file_path = Path(jsonl_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training dataset file '{jsonl_path}' does not exist")

        # 1. Upload training file to OpenAI
        with open(file_path, "rb") as f:
            file_response = await self.client.files.create(
                file=f,
                purpose="fine-tune",
            )
        file_id = file_response.id

        # 2. Create fine-tuning job
        job_response = await self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
        )

        provider_job_id = job_response.id
        logger.info("Submitted fine-tuning job %s with OpenAI ID %s", job_id, provider_job_id)
        return provider_job_id

    async def register_model_in_redis(
        self,
        redis_client,
        model_name: str,
        domain: str = "enterprise_rag",
        task_types: list[str] | None = None,
    ) -> None:
        """Register the fine-tuned model into Redis router:models."""
        if not redis_client:
            return

        task_types = task_types or ["rag_generation", "factual", "synthesis", "generation"]

        new_entry = {
            "model": model_name,
            "provider": "openai",
            "is_fine_tuned": True,
            "task_types": task_types,
            "domain": domain,
            "context_window": 128000,
            "estimated_latency_ms": 500,
            "estimated_cost_per_call": 0.002,
            "supports_vision": False,
            "is_judge": False,
        }

        try:
            raw = await redis_client.get("router:models")
            models: list[dict[str, Any]] = []
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                models = json.loads(raw)

            # Update if existing, otherwise append
            updated = False
            for i, m in enumerate(models):
                if m.get("model") == model_name:
                    models[i] = new_entry
                    updated = True
                    break
            if not updated:
                models.append(new_entry)

            await redis_client.set("router:models", json.dumps(models))
            logger.info("Successfully registered fine-tuned model '%s' in Redis router:models", model_name)
        except Exception as exc:
            logger.error("Failed to register fine-tuned model '%s' in Redis: %s", model_name, exc)

    async def poll_job_status(
        self,
        job_id: UUID | str,
        db_pool,
        redis_client=None,
    ) -> dict[str, Any]:
        """Poll job status from OpenAI, update database on completion, and register fine-tuned model."""
        job_uuid = UUID(str(job_id))

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, provider_job_id, base_model, status, mlflow_run_id FROM finetune_jobs WHERE id = $1",
                job_uuid,
            )
            if not row:
                return {"status": "not_found"}

            provider_job_id = row["provider_job_id"]
            if not provider_job_id:
                return {"status": row["status"]}

            # If already terminal, return
            if row["status"] in ("succeeded", "failed", "cancelled"):
                return {"status": row["status"]}

            # 1. Retrieve job status from OpenAI
            try:
                job_info = await self.client.fine_tuning.jobs.retrieve(provider_job_id)
                status = job_info.status
            except Exception as exc:
                logger.warning("Error retrieving fine-tuning status for job %s: %s", provider_job_id, exc)
                return {"status": "polling_error", "error": str(exc)}

            # 2. Handle completion
            if status == "succeeded":
                fine_tuned_model = getattr(job_info, "fine_tuned_model", None)
                trained_tokens = getattr(job_info, "trained_tokens", None)

                metrics_data = {}
                if trained_tokens is not None:
                    metrics_data["trained_tokens"] = trained_tokens
                if fine_tuned_model:
                    metrics_data["fine_tuned_model"] = fine_tuned_model

                # Update database
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'succeeded',
                        fine_tuned_model = $1,
                        metrics = $2,
                        completed_at = NOW()
                    WHERE id = $3
                    """,
                    fine_tuned_model,
                    json.dumps(metrics_data) if metrics_data else None,
                    job_uuid,
                )

                # Mark included_in_job for training pairs
                await conn.execute(
                    """
                    UPDATE training_pairs
                    SET included_in_job = $1
                    WHERE included_in_job IS NULL
                    """,
                    job_uuid,
                )

                # Register in Redis router
                if redis_client and fine_tuned_model:
                    await self.register_model_in_redis(redis_client, fine_tuned_model)

                # Log completion metrics in MLflow
                if row["mlflow_run_id"]:
                    await self.tracker.log_completion_metrics(
                        mlflow_run_id=row["mlflow_run_id"],
                        training_loss=getattr(job_info, "training_loss", None),
                        validation_loss=getattr(job_info, "validation_loss", None),
                        training_token_count=trained_tokens,
                        model_name=fine_tuned_model,
                    )

                return {
                    "status": "succeeded",
                    "fine_tuned_model": fine_tuned_model,
                }

            elif status in ("failed", "cancelled"):
                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = $1,
                        completed_at = NOW()
                    WHERE id = $2
                    """,
                    status,
                    job_uuid,
                )
                return {"status": status}

            else:
                # Still running or validating
                return {"status": status}
