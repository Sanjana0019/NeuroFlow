import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api.finetune import router as finetune_router
from backend.providers.router import ModelRouter, RoutingCriteria
from pipelines.finetuning.extractor import TrainingDataExtractor
from pipelines.finetuning.job_manager import FineTuningJobManager
from pipelines.finetuning.tracker import FineTuningTracker


# ============================================================================
# 1. Extractor Unit Tests (Quality, PII, Citations, Tokens, Faithfulness)
# ============================================================================


def test_extractor_token_count_validation(tmp_path):
    """Test token length constraints (50 - 2000 tokens)."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)

    # 1. Under 50 tokens -> Fail
    short_text = "According to [Source 1], the answer is brief."
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Explain something",
        assistant_message=short_text,
        faithfulness=0.9,
    )
    assert not is_valid
    assert "too short" in reason

    # 2. Between 50 and 2000 tokens with [Source 1] -> Valid
    valid_text = (
        "According to [Source 1], neural networks use backpropagation to optimize weights across layers. "
        * 6
    )
    assert extractor.count_tokens(valid_text) >= 50
    assert extractor.count_tokens(valid_text) <= 2000

    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Explain backpropagation",
        assistant_message=valid_text,
        faithfulness=0.95,
    )
    assert is_valid
    assert reason is None

    # 3. Over 2000 tokens -> Fail
    long_text = "[Source 1] " + ("supercalifragilisticexpialidocious word repetition test " * 500)
    assert extractor.count_tokens(long_text) > 2000
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Explain something long",
        assistant_message=long_text,
        faithfulness=0.95,
    )
    assert not is_valid
    assert "too long" in reason


def test_extractor_citation_requirement(tmp_path):
    """Test that responses without [Source N] are rejected."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)
    text_without_citation = "This is a detailed and well-written response that spans more than fifty words in length, but unfortunately does not cite any source brackets like Source 1 or Document 1 anywhere in the content." * 2

    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Summarize the contract",
        assistant_message=text_without_citation,
        faithfulness=0.95,
    )
    assert not is_valid
    assert "missing required [Source N] citation" in reason


def test_extractor_faithfulness_threshold(tmp_path):
    """Test faithfulness threshold (> 0.8)."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)
    good_text = (
        "Based on [Source 1], the indemnity clause is enforceable under Delaware state law as defined in section 4. "
        * 6
    )

    # 1. Faithfulness <= 0.8 -> Reject
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Is indemnity enforceable?",
        assistant_message=good_text,
        faithfulness=0.79,
    )
    assert not is_valid
    assert "Faithfulness score 0.79 is not > 0.8" in reason

    # 2. Faithfulness > 0.8 -> Pass
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Is indemnity enforceable?",
        assistant_message=good_text,
        faithfulness=0.85,
    )
    assert is_valid


def test_extractor_pii_filtering(tmp_path):
    """Test rejection of user queries containing email and phone PII."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)
    good_text = "According to [Source 1], the process is described in chapter 2. " * 6

    # 1. Email PII in query
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Send the details to john.doe@example.com immediately",
        assistant_message=good_text,
        faithfulness=0.9,
    )
    assert not is_valid
    assert "PII" in reason

    # 2. Phone PII in query
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="Call me at 555-123-4567 regarding the agreement",
        assistant_message=good_text,
        faithfulness=0.9,
    )
    assert not is_valid
    assert "PII" in reason

    # 3. Clean query
    is_valid, reason = extractor.validate_pair(
        system_prompt="System",
        user_message="What is the standard termination clause?",
        assistant_message=good_text,
        faithfulness=0.9,
    )
    assert is_valid


# ============================================================================
# 2. Mock Database & Client Setup for End-to-End Task 9 Testing
# ============================================================================


class MockTask9DBConn:
    def __init__(self):
        self.training_pairs: list[dict] = []
        self.pipeline_runs: list[dict] = []
        self.finetune_jobs: dict[UUID, dict] = {}

    def transaction(self):
        class Tx:
            async def __aenter__(self):
                pass
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return Tx()

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split())

        if "SELECT id, provider_job_id, base_model, status, mlflow_run_id FROM finetune_jobs WHERE id =" in normalized:
            job_id = UUID(str(args[0]))
            return self.finetune_jobs.get(job_id)

        if "SELECT id, provider_job_id, base_model, fine_tuned_model, status" in normalized and "FROM finetune_jobs WHERE id =" in normalized:
            job_id = UUID(str(args[0]))
            return self.finetune_jobs.get(job_id)

        if "UPDATE finetune_jobs SET provider_job_id =" in normalized:
            p_job_id, mlflow_id, j_id = args[0], args[1], UUID(str(args[2]))
            if j_id in self.finetune_jobs:
                self.finetune_jobs[j_id]["provider_job_id"] = p_job_id
                self.finetune_jobs[j_id]["mlflow_run_id"] = mlflow_id
                self.finetune_jobs[j_id]["status"] = "running"
                return self.finetune_jobs[j_id]

        return None

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())

        if "FROM pipeline_runs r1" in normalized and "JOIN pipeline_runs r2" in normalized:
            min_chosen = args[0] if len(args) > 0 else 4
            max_rejected = args[1] if len(args) > 1 else 2
            runs_by_query: dict[str, list[dict]] = {}
            for r in self.pipeline_runs:
                q_key = (r.get("query") or "").strip().lower()
                if q_key not in runs_by_query:
                    runs_by_query[q_key] = []
                runs_by_query[q_key].append(r)

            rows = []
            for q_key, q_runs in runs_by_query.items():
                good_runs = [r for r in q_runs if r.get("user_rating", 0) >= min_chosen]
                bad_runs = [r for r in q_runs if r.get("user_rating", 5) <= max_rejected]
                for gr in good_runs:
                    for br in bad_runs:
                        if gr["id"] != br["id"]:
                            rows.append({
                                "prompt": gr["query"],
                                "chosen": gr["generation"],
                                "chosen_rating": gr.get("user_rating"),
                                "rejected": br["generation"],
                                "rejected_rating": br.get("user_rating"),
                            })
            return rows

        if "FROM training_pairs tp" in normalized:
            rows = []
            for tp in self.training_pairs:
                u_rating = tp.get("user_rating")
                rows.append({
                    "id": tp["id"],
                    "run_id": tp["run_id"],
                    "system_prompt": tp.get("system_prompt", "You are a precise research assistant."),
                    "user_message": tp["user_message"],
                    "assistant_message": tp["assistant_message"],
                    "quality_score": tp.get("quality_score", 0.9),
                    "faithfulness": tp.get("faithfulness", 0.95),
                    "answer_relevance": tp.get("answer_relevance", 0.90),
                    "context_precision": tp.get("context_precision", 0.85),
                    "context_recall": tp.get("context_recall", 0.88),
                    "user_rating": u_rating,
                    "included_in_job": tp.get("included_in_job"),
                    "created_at": tp.get("created_at", datetime.now(timezone.utc)),
                    "query": tp["user_message"],
                    "generation": tp["assistant_message"],
                    "retrieved_chunk_ids": [],
                })
            return rows

        if "FROM finetune_jobs" in normalized:
            return list(self.finetune_jobs.values())

        return []

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())

        if "INSERT INTO finetune_jobs" in query:
            j_id, base_m, count = UUID(str(args[0])), args[1], args[2]
            now = datetime.now(timezone.utc)
            self.finetune_jobs[j_id] = {
                "id": j_id,
                "provider_job_id": None,
                "base_model": base_m,
                "fine_tuned_model": None,
                "status": "pending",
                "training_pair_count": count,
                "mlflow_run_id": None,
                "metrics": None,
                "created_at": now,
                "completed_at": None,
            }

        if "UPDATE finetune_jobs SET status = 'succeeded'" in normalized:
            ft_model, metrics_json, j_id = args[0], args[1], UUID(str(args[2]))
            if j_id in self.finetune_jobs:
                self.finetune_jobs[j_id]["status"] = "succeeded"
                self.finetune_jobs[j_id]["fine_tuned_model"] = ft_model
                self.finetune_jobs[j_id]["metrics"] = metrics_json
                self.finetune_jobs[j_id]["completed_at"] = datetime.now(timezone.utc)

        if "UPDATE training_pairs SET included_in_job =" in normalized:
            j_id = UUID(str(args[0]))
            for tp in self.training_pairs:
                if tp.get("included_in_job") is None:
                    tp["included_in_job"] = j_id


class MockTask9DBPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        class Ctx:
            def __init__(self, c):
                self.c = c
            async def __aenter__(self):
                return self.c
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return Ctx(self.conn)


class MockRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value

    async def aclose(self):
        pass


def seed_training_pairs(db_conn: MockTask9DBConn, count: int = 15):
    """Seed eligible and non-eligible training pairs."""
    for i in range(count):
        valid_assistant_msg = (
            f"According to [Source 1], the terms of clause {i+1} require mutual indemnification between parties. "
            * 5
        )
        db_conn.training_pairs.append({
            "id": uuid4(),
            "run_id": uuid4(),
            "system_prompt": "You are a precise research assistant.",
            "user_message": f"What are the terms of clause {i+1} in the agreement?",
            "assistant_message": valid_assistant_msg,
            "quality_score": 0.88,
            "faithfulness": 0.95,
            "user_rating": 5,
            "included_in_job": None,
            "created_at": datetime.now(timezone.utc),
        })


# ============================================================================
# 3. MLflow Tracking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_mlflow_tracker_logging(tmp_path):
    """Test MLflow tracker run creation, parameter logging, and completion metrics with mocked HTTP."""
    tracker = FineTuningTracker(tracking_uri="http://localhost:5000")
    dummy_jsonl = tmp_path / "test_data.jsonl"
    dummy_jsonl.write_text('{"messages": []}\n')

    with patch("httpx.AsyncClient.get") as mock_get, patch("httpx.AsyncClient.post") as mock_post:
        # Mock experiment response
        mock_get.return_value = MagicMock(is_success=True, json=lambda: {"experiment": {"experiment_id": "101"}})
        mock_post.return_value = MagicMock(is_success=True, json=lambda: {"run": {"info": {"run_id": "run-xyz-123"}}})

        run_id = await tracker.create_job_run(
            job_id="job-456",
            base_model="gpt-4o-mini-2024-07-18",
            training_pair_count=25,
            avg_quality_score=0.89,
            date_range="2026-08-01 to 2026-08-20",
            jsonl_path=str(dummy_jsonl),
        )

        assert run_id == "run-xyz-123"
        assert mock_post.call_count >= 5  # run create + 5 parameter logs

        # Log completion metrics & model registration
        await tracker.log_completion_metrics(
            mlflow_run_id=run_id,
            training_loss=0.12,
            validation_loss=0.14,
            training_token_count=15000,
            model_name="ft:gpt-4o-mini-neuroflow:v1",
        )
        assert mock_post.call_count > 6


# ============================================================================
# 4. OpenAI Fine-Tuning Job Manager & Redis Model Registration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_job_manager_submission_and_completion(tmp_path):
    """Test OpenAI fine-tune job submission, status polling, and Redis ModelRouter registration."""
    mock_openai = AsyncMock()
    mock_openai.files.create.return_value = MagicMock(id="file-openai-999")
    mock_openai.fine_tuning.jobs.create.return_value = MagicMock(id="ftjob-openai-888")

    tracker = FineTuningTracker()
    manager = FineTuningJobManager(openai_client=mock_openai, tracker=tracker)

    dummy_jsonl = tmp_path / "test.jsonl"
    dummy_jsonl.write_text('{"messages": []}\n')

    # 1. Submit job
    provider_id = await manager.submit_job(
        job_id="my-job-1",
        jsonl_path=dummy_jsonl,
        base_model="gpt-4o-mini-2024-07-18",
    )
    assert provider_id == "ftjob-openai-888"
    mock_openai.files.create.assert_called_once()
    mock_openai.fine_tuning.jobs.create.assert_called_once()

    # 2. Poll job status -> Succeeded
    db_conn = MockTask9DBConn()
    db_pool = MockTask9DBPool(db_conn)
    redis_mock = MockRedis()
    seed_training_pairs(db_conn, count=5)

    job_uuid = uuid4()
    await db_conn.execute(f"INSERT INTO finetune_jobs", job_uuid, "gpt-4o-mini", 5)
    db_conn.finetune_jobs[job_uuid]["provider_job_id"] = "ftjob-openai-888"

    mock_openai.fine_tuning.jobs.retrieve.return_value = MagicMock(
        status="succeeded",
        fine_tuned_model="ft:gpt-4o-mini-2024-07-18:neuroflow-custom:001",
        trained_tokens=12500,
    )

    poll_result = await manager.poll_job_status(
        job_id=job_uuid,
        db_pool=db_pool,
        redis_client=redis_mock,
    )

    assert poll_result["status"] == "succeeded"
    assert poll_result["fine_tuned_model"] == "ft:gpt-4o-mini-2024-07-18:neuroflow-custom:001"
    assert db_conn.finetune_jobs[job_uuid]["status"] == "succeeded"

    # Verify model is registered in Redis router:models
    raw_models = await redis_mock.get("router:models")
    assert raw_models is not None
    models_list = json.loads(raw_models)
    assert len(models_list) == 1
    assert models_list[0]["model"] == "ft:gpt-4o-mini-2024-07-18:neuroflow-custom:001"
    assert models_list[0]["is_fine_tuned"] is True
    assert "rag_generation" in models_list[0]["task_types"]

    # Verify ModelRouter routes to fine-tuned model when prefer_fine_tuned=True
    router = ModelRouter(redis=redis_mock)
    selected = await router.route(RoutingCriteria(task_type="rag_generation", prefer_fine_tuned=True))
    assert selected["model"] == "ft:gpt-4o-mini-2024-07-18:neuroflow-custom:001"


# ============================================================================
# 5. Fine-Tuning API Endpoints Tests
# ============================================================================


@pytest.fixture
def task9_client_and_db(tmp_path):
    app = FastAPI()
    app.include_router(finetune_router)

    db_conn = MockTask9DBConn()
    db_pool = MockTask9DBPool(db_conn)

    app.state.db_pool = db_pool
    app.state.neuroflow_client = None
    app.state.arq_redis = None

    test_client = TestClient(app)
    return test_client, db_conn


def test_preview_training_data_endpoint(task9_client_and_db):
    """GET /finetune/training-data/preview returns up to 5 eligible pairs."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=8)

    res = client.get("/finetune/training-data/preview")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 5
    assert "messages" in data[0]
    assert len(data[0]["messages"]) == 3
    assert data[0]["messages"][1]["role"] == "user"
    assert data[0]["messages"][2]["role"] == "assistant"


def test_create_finetune_job_insufficient_pairs(task9_client_and_db):
    """POST /finetune/jobs rejects when less than 10 eligible pairs exist."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=4)  # Only 4 pairs

    with patch("backend.api.finetune.settings.openai_api_key", "test-openai-key"):
        res = client.post("/finetune/jobs", json={"base_model": "gpt-4o-mini-2024-07-18"})
        assert res.status_code == 400
        assert "Insufficient training pairs" in res.json()["detail"]


def test_create_and_get_finetune_job_lifecycle(task9_client_and_db):
    """POST /finetune/jobs submits fine-tuning job and GET returns status."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=15)  # 15 pairs (>= 10 required)

    with patch("backend.api.finetune.settings.openai_api_key", "test-openai-key"), \
         patch("pipelines.finetuning.tracker.FineTuningTracker.create_job_run", new_callable=AsyncMock) as mock_tracker, \
         patch("pipelines.finetuning.job_manager.FineTuningJobManager.submit_job", new_callable=AsyncMock) as mock_submit:

        mock_tracker.return_value = "mlflow-run-abc"
        mock_submit.return_value = "ftjob-openai-12345"

        # 1. Create Job
        res = client.post(
            "/finetune/jobs",
            json={"base_model": "gpt-4o-mini-2024-07-18", "min_quality_score": 0.82},
        )
        assert res.status_code == 201
        data = res.json()
        job_id = data["job_id"]
        assert data["status"] == "running"
        assert data["provider_job_id"] == "ftjob-openai-12345"
        assert data["training_pair_count"] == 15
        assert data["mlflow_run_id"] == "mlflow-run-abc"

        # 2. GET /finetune/jobs list
        list_res = client.get("/finetune/jobs")
        assert list_res.status_code == 200
        jobs_list = list_res.json()
        assert len(jobs_list) == 1
        assert jobs_list[0]["job_id"] == job_id

        # 3. GET /finetune/jobs/{id}
        get_res = client.get(f"/finetune/jobs/{job_id}")
        assert get_res.status_code == 200
        assert get_res.json()["job_id"] == job_id
        assert get_res.json()["status"] == "running"


# ============================================================================
# 6. DPO Preference Extraction Tests
# ============================================================================


@pytest.mark.asyncio
async def test_dpo_pair_extraction_matching_same_query(tmp_path):
    """Test extracting DPO preference pairs where good (>=4) and bad (<=2) responses exist for same query."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)
    db_conn = MockTask9DBConn()
    db_pool = MockTask9DBPool(db_conn)

    # Add good response (user_rating = 5) and bad response (user_rating = 1) for same query
    query_text = "What is the warranty period for hardware?"
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": query_text,
        "generation": "According to [Source 1], the warranty period is 3 years from the date of purchase.",
        "user_rating": 5,
    })
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": query_text,
        "generation": "I am not sure, check the website maybe.",
        "user_rating": 1,
    })

    # Add an unrelated query with only a good response
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "What is the return policy?",
        "generation": "According to [Source 2], returns are accepted within 30 days.",
        "user_rating": 5,
    })

    job_id = uuid4()
    dpo_pairs, output_file = await extractor.extract_dpo_pairs(db_pool=db_pool, job_id=job_id)

    assert len(dpo_pairs) == 1
    pair = dpo_pairs[0]
    assert pair["prompt"] == query_text
    assert "3 years" in pair["chosen"]
    assert "check the website" in pair["rejected"]

    # Verify JSONL file format
    assert output_file is not None and output_file.exists()
    content = output_file.read_text(encoding="utf-8").strip()
    loaded = json.loads(content)
    assert "prompt" in loaded
    assert "chosen" in loaded
    assert "rejected" in loaded
    assert loaded["prompt"] == query_text


@pytest.mark.asyncio
async def test_dpo_pair_extraction_missing_side(tmp_path):
    """Test no DPO pair is extracted when either good (rating >= 4) or bad (rating <= 2) is missing."""
    extractor = TrainingDataExtractor(training_data_dir=tmp_path)
    db_conn = MockTask9DBConn()
    db_pool = MockTask9DBPool(db_conn)

    # 1. Only good responses (rating 4 and rating 5) -> No DPO pair
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "What is the SLA?",
        "generation": "SLA is 99.99% uptime guaranteed.",
        "user_rating": 5,
    })
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "What is the SLA?",
        "generation": "SLA guarantees 99.9% uptime per agreement.",
        "user_rating": 4,
    })

    # 2. Only bad responses (rating 1 and rating 2) -> No DPO pair
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "How to reset password?",
        "generation": "Unknown error.",
        "user_rating": 1,
    })
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "How to reset password?",
        "generation": "Contact support.",
        "user_rating": 2,
    })

    # 3. Rating 3 (neutral) ignored
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "Where is the office?",
        "generation": "Office is in New York.",
        "user_rating": 5,
    })
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "Where is the office?",
        "generation": "Somewhere in USA.",
        "user_rating": 3,
    })

    dpo_pairs, _ = await extractor.extract_dpo_pairs(db_pool=db_pool)
    assert len(dpo_pairs) == 0


def test_preview_dpo_data_endpoint(task9_client_and_db):
    """GET /finetune/dpo/preview returns up to 5 DPO preference pairs."""
    client, db_conn = task9_client_and_db

    for i in range(3):
        q = f"Question number {i+1} regarding policy"
        db_conn.pipeline_runs.append({
            "id": uuid4(),
            "query": q,
            "generation": f"Chosen detailed answer for question {i+1}",
            "user_rating": 5,
        })
        db_conn.pipeline_runs.append({
            "id": uuid4(),
            "query": q,
            "generation": f"Rejected brief answer for question {i+1}",
            "user_rating": 1,
        })

    res = client.get("/finetune/dpo/preview")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 3
    assert "prompt" in data[0]
    assert "chosen" in data[0]
    assert "rejected" in data[0]


def test_create_job_requires_openai_key(task9_client_and_db):
    """POST /finetune/jobs rejects with 400 when OPENAI_API_KEY is not configured."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=15)

    with patch("backend.api.finetune.settings.openai_api_key", None):
        res = client.post("/finetune/jobs", json={"base_model": "gpt-4o-mini-2024-07-18"})
        assert res.status_code == 400
        assert "OpenAI API key is not configured" in res.json()["detail"]


def test_dataset_readiness_endpoint(task9_client_and_db):
    """GET /finetune/readiness returns canonical readiness breakdown and validation rules."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=3)

    res = client.get("/finetune/readiness")
    assert res.status_code == 200
    data = res.json()

    assert data["total_candidates"] == 3
    assert data["eligible_sft_count"] == 3
    assert data["min_required_for_finetuning"] == 10
    assert data["remaining_for_finetuning"] == 7
    assert data["can_export"] is True
    assert len(data["validation_rules"]) >= 4


def test_training_data_list_endpoint(task9_client_and_db):
    """GET /finetune/training-data returns all training pairs with validation and token details."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=4)

    res = client.get("/finetune/training-data")
    assert res.status_code == 200
    data = res.json()

    assert len(data) == 4
    first = data[0]
    assert "id" in first
    assert "user_message" in first
    assert "assistant_message" in first
    assert "token_count" in first
    assert first["token_count"] >= 50
    assert first["has_citation"] is True
    assert first["is_valid"] is True


def test_export_dataset_sft_jsonl_endpoint(task9_client_and_db):
    """GET /finetune/datasets/export?dataset_type=sft&format=jsonl returns valid JSONL format."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=5)

    res = client.get("/finetune/datasets/export?dataset_type=sft&format=jsonl")
    assert res.status_code == 200
    assert "application/x-jsonlines" in res.headers["content-type"]
    assert "attachment" in res.headers["content-disposition"]
    assert res.headers["X-Total-Records"] == "5"

    lines = res.text.strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        obj = json.loads(line)
        assert "messages" in obj
        assert len(obj["messages"]) == 3
        assert obj["messages"][0]["role"] == "system"
        assert obj["messages"][1]["role"] == "user"
        assert obj["messages"][2]["role"] == "assistant"


def test_export_dataset_sft_json_endpoint(task9_client_and_db):
    """GET /finetune/datasets/export?dataset_type=sft&format=json returns valid JSON array."""
    client, db_conn = task9_client_and_db
    seed_training_pairs(db_conn, count=3)

    res = client.get("/finetune/datasets/export?dataset_type=sft&format=json")
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert "messages" in data[0]


def test_export_dataset_dpo_jsonl_endpoint(task9_client_and_db):
    """GET /finetune/datasets/export?dataset_type=dpo&format=jsonl returns valid DPO JSONL format."""
    client, db_conn = task9_client_and_db

    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "What is the return policy?",
        "generation": "Returns accepted within 30 days per policy.",
        "user_rating": 5,
    })
    db_conn.pipeline_runs.append({
        "id": uuid4(),
        "query": "What is the return policy?",
        "generation": "Check elsewhere.",
        "user_rating": 1,
    })

    res = client.get("/finetune/datasets/export?dataset_type=dpo&format=jsonl")
    assert res.status_code == 200
    lines = res.text.strip().split("\n")
    assert len(lines) == 1
    dpo_obj = json.loads(lines[0])
    assert dpo_obj["prompt"] == "What is the return policy?"
    assert "within 30 days" in dpo_obj["chosen"]
    assert "Check elsewhere" in dpo_obj["rejected"]


