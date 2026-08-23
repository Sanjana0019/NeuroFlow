import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import httpx

try:
    from backend.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("neuroflow.finetuning.tracker")


class FineTuningTracker:
    """Tracks fine-tuning jobs, artifacts, parameters, metrics, and registered models in MLflow."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "neuroflow-finetuning"):
        self.tracking_uri = (tracking_uri or settings.mlflow_tracking_uri).rstrip("/")
        self.experiment_name = experiment_name
        self.experiment_id: str | None = None

    async def _ensure_experiment(self, client: httpx.AsyncClient) -> str:
        """Ensure the MLflow experiment exists and return its ID."""
        if self.experiment_id:
            return self.experiment_id

        try:
            # 1. Check if experiment exists
            res = await client.get(
                f"{self.tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": self.experiment_name},
                timeout=5.0,
            )
            if res.is_success:
                data = res.json()
                if "experiment" in data and "experiment_id" in data["experiment"]:
                    self.experiment_id = str(data["experiment"]["experiment_id"])
                    return self.experiment_id

            # 2. Create experiment
            create_res = await client.post(
                f"{self.tracking_uri}/api/2.0/mlflow/experiments/create",
                json={"name": self.experiment_name},
                timeout=5.0,
            )
            if create_res.is_success:
                data = create_res.json()
                self.experiment_id = str(data.get("experiment_id", "0"))
                return self.experiment_id
        except Exception as exc:
            logger.warning("MLflow experiment resolution error: %s", exc)

        return "0"

    async def create_job_run(
        self,
        job_id: str,
        base_model: str,
        training_pair_count: int,
        avg_quality_score: float,
        date_range: str,
        jsonl_path: str | Path | None = None,
    ) -> str:
        """Create an MLflow run, log parameters, and record dataset metadata/artifacts."""
        run_id = f"mlflow-{uuid4().hex[:12]}"
        now_ms = int(time.time() * 1000)

        params_to_log = {
            "job_id": str(job_id),
            "base_model": base_model,
            "training_pair_count": str(training_pair_count),
            "avg_quality_score": str(round(avg_quality_score, 4)),
            "date_range": str(date_range),
        }

        try:
            async with httpx.AsyncClient() as client:
                exp_id = await self._ensure_experiment(client)
                create_res = await client.post(
                    f"{self.tracking_uri}/api/2.0/mlflow/runs/create",
                    json={
                        "experiment_id": exp_id,
                        "start_time": now_ms,
                        "tags": [
                            {"key": "mlflow.runName", "value": f"finetune-{job_id}"},
                            {"key": "job_id", "value": str(job_id)},
                        ],
                    },
                    timeout=5.0,
                )
                if create_res.is_success:
                    run_data = create_res.json().get("run", {})
                    info = run_data.get("info", {})
                    run_id = info.get("run_id", run_id)

                # Log parameters
                for k, v in params_to_log.items():
                    await client.post(
                        f"{self.tracking_uri}/api/2.0/mlflow/runs/log-parameter",
                        json={"run_id": run_id, "key": k, "value": str(v)},
                        timeout=5.0,
                    )

                # Log artifact path note if file exists
                if jsonl_path and Path(jsonl_path).exists():
                    await client.post(
                        f"{self.tracking_uri}/api/2.0/mlflow/runs/log-parameter",
                        json={"run_id": run_id, "key": "dataset_artifact", "value": str(jsonl_path)},
                        timeout=5.0,
                    )
        except Exception as exc:
            logger.warning("MLflow run creation warning: %s (using local run_id %s)", exc, run_id)

        return run_id

    async def log_completion_metrics(
        self,
        mlflow_run_id: str,
        training_loss: float | None = None,
        validation_loss: float | None = None,
        training_token_count: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Log post-training loss/tokens metrics, register model in MLflow, and mark run finished."""
        now_ms = int(time.time() * 1000)

        metrics = {}
        if training_loss is not None:
            metrics["training_loss"] = float(training_loss)
        if validation_loss is not None:
            metrics["validation_loss"] = float(validation_loss)
        if training_token_count is not None:
            metrics["training_token_count"] = float(training_token_count)

        try:
            async with httpx.AsyncClient() as client:
                # Log metrics
                for k, v in metrics.items():
                    await client.post(
                        f"{self.tracking_uri}/api/2.0/mlflow/runs/log-metric",
                        json={"run_id": mlflow_run_id, "key": k, "value": v, "timestamp": now_ms},
                        timeout=5.0,
                    )

                # Register model in MLflow Model Registry if provided
                if model_name:
                    try:
                        # 1. Create registered model if not exists
                        await client.post(
                            f"{self.tracking_uri}/api/2.0/mlflow/registered-models/create",
                            json={"name": model_name},
                            timeout=5.0,
                        )
                    except Exception:
                        pass

                    try:
                        # 2. Create model version
                        await client.post(
                            f"{self.tracking_uri}/api/2.0/mlflow/model-versions/create",
                            json={
                                "name": model_name,
                                "source": f"runs:/{mlflow_run_id}/model",
                                "run_id": mlflow_run_id,
                            },
                            timeout=5.0,
                        )
                    except Exception as exc:
                        logger.warning("MLflow model version registration note: %s", exc)

                # Mark run finished
                await client.post(
                    f"{self.tracking_uri}/api/2.0/mlflow/runs/update",
                    json={"run_id": mlflow_run_id, "status": "FINISHED", "end_time": now_ms},
                    timeout=5.0,
                )
        except Exception as exc:
            logger.warning("MLflow metric logging warning: %s", exc)
