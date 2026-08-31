import asyncio
import copy
import json
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api.analytics_utils import calculate_percentile, calculate_run_cost
from backend.api.compare import router as compare_router
from backend.api.pipelines import router as pipelines_router
from backend.api.query import router as query_router
from backend.models.pipeline import (
    EvaluationConfig,
    GenerationConfig,
    IngestionConfig,
    ModelRoutingConfig,
    PipelineConfig,
    RetrievalConfig,
)


def get_valid_pipeline_config_dict(name="enterprise_rag_v1"):
    return {
        "name": name,
        "description": "Production enterprise retrieval-augmented generation pipeline",
        "ingestion": {
            "chunking_strategy": "recursive",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "extractors_enabled": ["pdf", "docx", "text"],
        },
        "retrieval": {
            "dense_k": 20,
            "sparse_k": 20,
            "reranker": "bge-reranker-large",
            "top_k_after_rerank": 5,
            "query_expansion": True,
            "metadata_filters_enabled": True,
        },
        "generation": {
            "model_routing": {
                "task_type": "rag_generation",
                "max_cost_per_call": 0.05,
            },
            "max_context_tokens": 4000,
            "temperature": 0.7,
            "system_prompt_variant": "enterprise_qa",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.85,
        },
    }


# ============================================================================
# 1. Pydantic PipelineConfig Validation Tests
# ============================================================================


def test_pipeline_config_valid():
    """Valid configuration parses successfully."""
    raw = get_valid_pipeline_config_dict()
    cfg = PipelineConfig.model_validate(raw)
    assert cfg.name == "enterprise_rag_v1"
    assert cfg.ingestion.chunk_size_tokens == 512
    assert cfg.retrieval.dense_k == 20
    assert cfg.generation.temperature == 0.7
    assert cfg.evaluation.training_threshold == 0.85


def test_pipeline_config_unknown_top_level_key_fails():
    """Unknown top-level keys must raise ValidationError."""
    raw = get_valid_pipeline_config_dict()
    raw["unknown_extra_field"] = "malicious_payload"

    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(raw)
    assert "extra_forbidden" in str(exc_info.value)


def test_pipeline_config_unknown_nested_key_fails():
    """Unknown nested keys must raise ValidationError."""
    raw = get_valid_pipeline_config_dict()
    raw["retrieval"]["unsupported_algorithm"] = "quantum_search"

    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(raw)
    assert "extra_forbidden" in str(exc_info.value)

    raw2 = get_valid_pipeline_config_dict()
    raw2["generation"]["model_routing"]["unexpected_field"] = 123
    with pytest.raises(ValidationError) as exc_info2:
        PipelineConfig.model_validate(raw2)
    assert "extra_forbidden" in str(exc_info2.value)


def test_pipeline_config_invalid_constraints():
    """Invalid numeric constraints (negative tokens, temp > 2.0, threshold > 1.0) must fail."""
    # Chunk size <= 0
    raw = get_valid_pipeline_config_dict()
    raw["ingestion"]["chunk_size_tokens"] = 0
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)

    # Negative chunk overlap
    raw = get_valid_pipeline_config_dict()
    raw["ingestion"]["chunk_overlap_tokens"] = -5
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)

    # Temperature > 2.0
    raw = get_valid_pipeline_config_dict()
    raw["generation"]["temperature"] = 2.5
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)

    # Threshold > 1.0
    raw = get_valid_pipeline_config_dict()
    raw["evaluation"]["training_threshold"] = 1.5
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)

    # Negative cost
    raw = get_valid_pipeline_config_dict()
    raw["generation"]["model_routing"]["max_cost_per_call"] = -0.01
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_ui_preset_configs_valid():
    """Verify that all 3 UI presets (Balanced Hybrid, Fast Dense, Deep Research) strictly validate."""
    balanced_hybrid = {
        "name": "enterprise-hybrid-rag",
        "description": "Production enterprise retrieval-augmented generation pipeline with hybrid search.",
        "ingestion": {
            "chunking_strategy": "recursive",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "extractors_enabled": ["pdf", "docx", "text"],
        },
        "retrieval": {
            "dense_k": 20,
            "sparse_k": 15,
            "reranker": "bge-reranker-large",
            "top_k_after_rerank": 5,
            "query_expansion": True,
            "metadata_filters_enabled": True,
        },
        "generation": {
            "model_routing": {
                "task_type": "rag_generation",
                "max_cost_per_call": 0.05,
            },
            "max_context_tokens": 4000,
            "temperature": 0.7,
            "system_prompt_variant": "enterprise_qa",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.85,
        },
    }
    cfg1 = PipelineConfig.model_validate(balanced_hybrid)
    assert cfg1.name == "enterprise-hybrid-rag"
    assert cfg1.retrieval.reranker == "bge-reranker-large"

    fast_dense = {
        "name": "fast-dense-search",
        "description": "High-speed dense vector search pipeline for rapid question-answering.",
        "ingestion": {
            "chunking_strategy": "recursive",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 32,
            "extractors_enabled": ["pdf", "docx", "text"],
        },
        "retrieval": {
            "dense_k": 10,
            "sparse_k": 5,
            "reranker": "none",
            "top_k_after_rerank": 5,
            "query_expansion": False,
            "metadata_filters_enabled": False,
        },
        "generation": {
            "model_routing": {
                "task_type": "rag_generation",
                "max_cost_per_call": 0.02,
            },
            "max_context_tokens": 2048,
            "temperature": 0.3,
            "system_prompt_variant": "concise",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.80,
        },
    }
    cfg2 = PipelineConfig.model_validate(fast_dense)
    assert cfg2.name == "fast-dense-search"
    assert cfg2.retrieval.sparse_k == 5

    deep_research = {
        "name": "deep-research-rag",
        "description": "Comprehensive research pipeline with expanded context window and deep cross-encoder reranking.",
        "ingestion": {
            "chunking_strategy": "recursive",
            "chunk_size_tokens": 1024,
            "chunk_overlap_tokens": 128,
            "extractors_enabled": ["pdf", "docx", "text", "csv", "pptx"],
        },
        "retrieval": {
            "dense_k": 30,
            "sparse_k": 25,
            "reranker": "bge-reranker-large",
            "top_k_after_rerank": 10,
            "query_expansion": True,
            "metadata_filters_enabled": True,
        },
        "generation": {
            "model_routing": {
                "task_type": "rag_generation",
                "max_cost_per_call": 0.10,
            },
            "max_context_tokens": 8000,
            "temperature": 0.5,
            "system_prompt_variant": "comprehensive_research",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.90,
        },
    }
    cfg3 = PipelineConfig.model_validate(deep_research)
    assert cfg3.name == "deep-research-rag"
    assert cfg3.generation.max_context_tokens == 8000


# ============================================================================
# Mock Infrastructure for API Testing
# ============================================================================


class MockTask8DBConn:
    def __init__(self):
        self.pipelines: dict[UUID, dict] = {}
        self.pipeline_versions: list[dict] = []
        self.pipeline_runs: list[dict] = []
        self.evaluations: list[dict] = []

    def transaction(self):
        class Tx:
            async def __aenter__(self):
                pass
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return Tx()

    async def fetchrow(self, query: str, *args):
        if "INSERT INTO pipelines" in query:
            p_id, name, desc, cfg_json = args[0], args[1], args[2], args[3]
            p_uuid = UUID(str(p_id))
            now = datetime.now(timezone.utc)
            row = {
                "id": p_uuid,
                "name": name,
                "description": desc,
                "version": 1,
                "status": "active",
                "config": cfg_json,
                "created_at": now,
                "updated_at": now,
            }
            self.pipelines[p_uuid] = row
            return row

        p_uuid = None
        if args:
            try:
                p_uuid = UUID(str(args[0]))
            except Exception:
                pass

        normalized = " ".join(query.split())

        if "SELECT id, version FROM pipelines WHERE id =" in normalized:
            if p_uuid and p_uuid in self.pipelines:
                p = self.pipelines[p_uuid]
                return {"id": p["id"], "version": p["version"]}
            return None

        if "SELECT id, version, status FROM pipelines WHERE id =" in normalized:
            if p_uuid and p_uuid in self.pipelines:
                p = self.pipelines[p_uuid]
                return {"id": p["id"], "version": p["version"], "status": p["status"]}
            return None

        if "SELECT id, name, description, version, status, config, created_at, updated_at FROM pipelines WHERE id =" in normalized:
            if p_uuid and p_uuid in self.pipelines:
                return self.pipelines[p_uuid]
            return None

        if "FROM pipelines WHERE id =" in normalized:
            if p_uuid and p_uuid in self.pipelines:
                return self.pipelines[p_uuid]
            return None

        if "UPDATE pipelines SET name =" in normalized:
            name, desc, version, cfg_json, p_id = args[0], args[1], args[2], args[3], args[4]
            p_uuid = UUID(str(p_id))
            if p_uuid in self.pipelines:
                self.pipelines[p_uuid]["name"] = name
                self.pipelines[p_uuid]["description"] = desc
                self.pipelines[p_uuid]["version"] = version
                self.pipelines[p_uuid]["config"] = cfg_json
                self.pipelines[p_uuid]["updated_at"] = datetime.now(timezone.utc)
                return self.pipelines[p_uuid]
            return None

        if "UPDATE pipelines SET status = 'archived'" in normalized:
            if p_uuid and p_uuid in self.pipelines:
                self.pipelines[p_uuid]["status"] = "archived"
                self.pipelines[p_uuid]["updated_at"] = datetime.now(timezone.utc)
                return {"id": p_uuid, "status": "archived"}
            return None

        if "COUNT(CASE WHEN r.created_at >= $2 THEN 1 END) AS count_7d" in normalized:
            runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            evals = [e for e in self.evaluations if e.get("pipeline_id") == p_uuid or any(r["id"] == e.get("run_id") for r in runs)]
            
            avg_faith = sum(e["faithfulness"] for e in evals if e.get("faithfulness") is not None) / len(evals) if evals else (0.92 if runs else None)
            avg_quality = sum(e["overall_score"] for e in evals if e.get("overall_score") is not None) / len(evals) if evals else (0.89 if runs else None)
            
            return {
                "count_7d": len(runs),
                "count_prev_7d": 0,
                "quality_7d": avg_quality,
                "quality_all_time": avg_quality,
                "faith_7d": avg_faith,
                "faith_all_time": avg_faith,
                "faith_prev_7d": None,
                "total_runs": len(runs),
            }

        if "COUNT(r.id) AS total_runs" in normalized:
            runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            return {
                "total_runs": len(runs),
                "avg_faithfulness": 0.92 if runs else None,
                "avg_answer_relevance": 0.88 if runs else None,
                "avg_context_precision": 0.85 if runs else None,
                "avg_context_recall": 0.89 if runs else None,
                "avg_overall_score": 0.89 if runs else None,
            }

        return None

    async def fetchval(self, query: str, *args):
        if "SELECT id FROM pipelines WHERE name =" in query:
            name = args[0]
            for p in self.pipelines.values():
                if p["name"] == name and p["status"] != "archived":
                    return p["id"]
            return None

        p_uuid = None
        if args:
            try:
                p_uuid = UUID(str(args[0]))
            except Exception:
                pass

        if "percentile_cont(0.5) WITHIN GROUP" in query:
            runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            if not runs:
                return None
            lats = sorted([r.get("latency_ms", 120.0) for r in runs])
            mid = len(lats) // 2
            return float(lats[mid])

        if "SELECT COUNT(*) FROM pipeline_runs WHERE pipeline_id =" in query:
            return sum(1 for r in self.pipeline_runs if r["pipeline_id"] == p_uuid)

        if "INSERT INTO pipeline_runs" in query:
            p_id, p_ver, q_text, chunk_ids = args[0], args[1], args[2], args[3]
            run_id = uuid4()
            self.pipeline_runs.append({
                "id": run_id,
                "pipeline_id": UUID(str(p_id)),
                "pipeline_version": p_ver,
                "query": q_text,
                "retrieved_chunk_ids": chunk_ids,
                "generation": "Sample answer",
                "latency_ms": 120,
                "input_tokens": 100,
                "output_tokens": 50,
                "model_used": "gpt-4o-mini",
                "status": "complete",
                "created_at": datetime.now(timezone.utc),
            })
            return run_id

        return None

    async def fetch(self, query: str, *args):
        p_uuid = None
        if args:
            try:
                p_uuid = UUID(str(args[0]))
            except Exception:
                pass

        if "SELECT DATE(created_at AT TIME ZONE 'UTC') AS day" in query:
            runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            now_day = datetime.now(timezone.utc).date()
            if runs:
                return [{"day": now_day, "count": len(runs)}]
            return []

        if "FROM pipelines p" in query:
            include_archived = "WHERE p.status != 'archived'" not in query
            rows = []
            for p in self.pipelines.values():
                if not include_archived and p["status"] == "archived":
                    continue
                runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p["id"]]
                rows.append({
                    "id": p["id"],
                    "name": p["name"],
                    "description": p["description"],
                    "version": p["version"],
                    "status": p["status"],
                    "config": p["config"],
                    "created_at": p["created_at"],
                    "updated_at": p["updated_at"],
                    "total_runs": len(runs),
                    "last_run_at": runs[-1]["created_at"] if runs else None,
                    "last_run_status": runs[-1]["status"] if runs else None,
                    "last_run_latency_ms": runs[-1]["latency_ms"] if runs else None,
                })
            return rows

        p_uuid = None
        if args:
            try:
                p_uuid = UUID(str(args[0]))
            except Exception:
                pass

        if "SELECT r.id AS run_id" in query:
            limit = args[1]
            offset = args[2]
            matching_runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            sliced = matching_runs[offset: offset + limit]
            rows = []
            for r in sliced:
                rows.append({
                    "run_id": r["id"],
                    "pipeline_id": r["pipeline_id"],
                    "pipeline_version": r.get("pipeline_version", 1),
                    "query": r["query"],
                    "generation": r.get("generation", "test response"),
                    "latency_ms": r.get("latency_ms", 150),
                    "input_tokens": r.get("input_tokens", 80),
                    "output_tokens": r.get("output_tokens", 40),
                    "model_used": r.get("model_used", "gpt-4o-mini"),
                    "status": r.get("status", "complete"),
                    "created_at": r["created_at"],
                    "faithfulness": 0.95,
                    "answer_relevance": 0.90,
                    "context_precision": 0.85,
                    "context_recall": 0.88,
                    "overall_score": 0.91,
                })
            return rows

        if "SELECT r.id, r.latency_ms" in query:
            matching_runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            rows = []
            for r in matching_runs:
                rows.append({
                    "id": r["id"],
                    "latency_ms": r.get("latency_ms", 120),
                    "input_tokens": r.get("input_tokens", 100),
                    "output_tokens": r.get("output_tokens", 50),
                    "model_used": r.get("model_used", "gpt-4o-mini"),
                    "created_at": r["created_at"],
                    "faithfulness": 0.95,
                    "answer_relevance": 0.90,
                    "context_precision": 0.85,
                    "context_recall": 0.88,
                    "overall_score": 0.91,
                })
            return rows

        if "status = 'failed'" in query:
            failed_runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid and r.get("status") == "failed"]
            return [
                {
                    "run_id": r["id"],
                    "query": r["query"],
                    "timestamp": r["created_at"],
                    "error_message": r.get("generation") or "Execution failed",
                }
                for r in failed_runs[:5]
            ]

        if "SELECT DATE(created_at) AS run_date" in query:
            today = datetime.now(timezone.utc).date()
            runs = [r for r in self.pipeline_runs if r["pipeline_id"] == p_uuid]
            return [{"run_date": today, "count": len(runs)}] if runs else []

        # Chunks for retrieval
        return [
            {
                "id": uuid4(),
                "document_id": uuid4(),
                "content": "Master Services Agreement Clause 14: Liability cap.",
                "chunk_index": 0,
                "token_count": 10,
                "metadata": json.dumps({"page_number": 1}),
                "filename": "msa.pdf",
                "score": 0.95,
            }
        ]

    async def execute(self, query: str, *args):
        if "INSERT INTO pipeline_versions" in query:
            p_id, ver, cfg_json = args[0], args[1], args[2]
            self.pipeline_versions.append({
                "pipeline_id": UUID(str(p_id)),
                "version": ver,
                "config": cfg_json,
                "created_at": datetime.now(timezone.utc),
            })
        if "UPDATE pipeline_runs" in query:
            pass


class MockTask8DBPool:
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


class MockTask8Provider:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.model = "gpt-4o-mini"

    async def complete(self, messages, **kwargs):
        system_content = messages[0].content if messages and hasattr(messages[0], "content") else ""
        if "query understanding" in str(system_content).lower() or "expanded_queries" in str(system_content).lower():
            content = json.dumps({
                "expanded_queries": ["liability clause in msa"],
                "metadata_filters": {},
                "query_type": "factual",
            })
        else:
            content = "Liability is capped at 12 months fees under Clause 14."

        return type(
            "Res",
            (),
            {
                "content": content,
                "model": "gpt-4o-mini",
                "input_tokens": 20,
                "output_tokens": 12,
                "cost_usd": 0.00002,
                "latency_ms": 50.0,
            },
        )()

    async def stream(self, messages, **kwargs):
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        yield "Liability "
        yield "is "
        yield "capped."

    async def embed(self, texts):
        return [[0.02] * 1536 for _ in texts]


class MockTask8Client:
    def __init__(self, delay=0.0):
        self.provider = MockTask8Provider(delay=delay)
        self.providers = {"openai": self.provider, "mock": self.provider}

    async def chat(self, messages, routing_criteria, **kwargs):
        return await self.provider.complete(messages)

    async def stream(self, messages, routing_criteria, **kwargs):
        return self.provider.stream(messages), "gpt-4o-mini"

    async def embed(self, texts):
        return await self.provider.embed(texts)


@pytest.fixture
def task8_client_and_db():
    app = FastAPI()
    app.include_router(pipelines_router)
    app.include_router(compare_router)
    app.include_router(query_router)

    db_conn = MockTask8DBConn()
    db_pool = MockTask8DBPool(db_conn)
    client_obj = MockTask8Client(delay=0.0)

    app.state.db_pool = db_pool
    app.state.neuroflow_client = client_obj
    app.state.arq_redis = None

    test_client = TestClient(app)
    return test_client, db_conn, app


# ============================================================================
# 2. Pipeline CRUD & Versioning Tests
# ============================================================================


def test_pipeline_crud_and_versioning(task8_client_and_db):
    """Test full CRUD lifecycle and historical version creation."""
    client, db, _ = task8_client_and_db

    # 1. Create pipeline v1
    cfg_raw = get_valid_pipeline_config_dict("test_pipeline_alpha")
    res = client.post("/pipelines", json=cfg_raw)
    assert res.status_code == 201
    data = res.json()
    pipeline_id = data["id"]
    assert data["version"] == 1
    assert data["status"] == "active"
    assert data["name"] == "test_pipeline_alpha"

    # Check version 1 stored in versions table
    assert len(db.pipeline_versions) == 1
    assert db.pipeline_versions[0]["version"] == 1

    # 2. Duplicate name creation fails
    res_dup = client.post("/pipelines", json=cfg_raw)
    assert res_dup.status_code == 400

    # 3. GET /pipelines/{id}
    res_get = client.get(f"/pipelines/{pipeline_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == pipeline_id
    assert res_get.json()["version"] == 1
    assert "evaluation_summary" in res_get.json()

    # 4. PATCH /pipelines/{id} -> Creates Version 2
    updated_cfg = copy.deepcopy(cfg_raw)
    updated_cfg["retrieval"]["dense_k"] = 35
    updated_cfg["generation"]["temperature"] = 0.2

    res_patch = client.patch(f"/pipelines/{pipeline_id}", json=updated_cfg)
    assert res_patch.status_code == 200
    patch_data = res_patch.json()
    assert patch_data["version"] == 2
    assert patch_data["config"]["retrieval"]["dense_k"] == 35

    # Check historical version preservation
    assert len(db.pipeline_versions) == 2
    v1_record = [v for v in db.pipeline_versions if v["version"] == 1][0]
    v2_record = [v for v in db.pipeline_versions if v["version"] == 2][0]
    assert json.loads(v1_record["config"])["retrieval"]["dense_k"] == 20
    assert json.loads(v2_record["config"])["retrieval"]["dense_k"] == 35

    # 5. GET /pipelines list includes metrics
    res_list = client.get("/pipelines")
    assert res_list.status_code == 200
    items = res_list.json()
    assert len(items) >= 1
    assert any(p["id"] == pipeline_id for p in items)

    # 6. Soft Delete /pipelines/{id}
    res_del = client.delete(f"/pipelines/{pipeline_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "archived"

    # Archived pipeline still preserved in DB
    assert UUID(pipeline_id) in db.pipelines
    assert db.pipelines[UUID(pipeline_id)]["status"] == "archived"


# ============================================================================
# 3. Pipeline Runs & Execution Integration
# ============================================================================


def test_pipeline_runs_and_version_recording(task8_client_and_db):
    """Ensure runs records the pipeline_version and pagination works."""
    client, db, _ = task8_client_and_db

    # Create pipeline
    cfg = get_valid_pipeline_config_dict("query_pipeline_test")
    p_res = client.post("/pipelines", json=cfg)
    p_id = p_res.json()["id"]

    # Execute query using this pipeline_id
    q_res = client.post(
        "/query",
        json={"query": "What is the liability cap?", "pipeline_id": p_id},
    )
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert "run_id" in q_data

    # Verify pipeline_runs recorded pipeline_version
    matching_runs = [r for r in db.pipeline_runs if str(r["pipeline_id"]) == str(p_id)]
    assert len(matching_runs) >= 1
    assert matching_runs[0]["pipeline_version"] == 1

    # Test GET /pipelines/{id}/runs with pagination
    runs_res = client.get(f"/pipelines/{p_id}/runs?page=1&page_size=10")
    assert runs_res.status_code == 200
    runs_data = runs_res.json()
    assert "items" in runs_data
    assert runs_data["total"] >= 1
    assert runs_data["page"] == 1
    assert runs_data["items"][0]["pipeline_version"] == 1


# ============================================================================
# 4. A/B Pipeline Comparison & Parallel Timing Verification
# ============================================================================


def test_ab_pipeline_comparison_concurrent_execution(task8_client_and_db):
    """Test /pipelines/compare executes both branches concurrently with timing verification."""
    client, db, app = task8_client_and_db

    # Create Pipeline A and Pipeline B
    cfg_a = get_valid_pipeline_config_dict("pipeline_a_fast")
    res_a = client.post("/pipelines", json=cfg_a)
    p_a_id = res_a.json()["id"]

    cfg_b = get_valid_pipeline_config_dict("pipeline_b_smart")
    cfg_b["retrieval"]["dense_k"] = 40
    res_b = client.post("/pipelines", json=cfg_b)
    p_b_id = res_b.json()["id"]

    # Introduce a 0.20s controlled delay in the mock client
    app.state.neuroflow_client = MockTask8Client(delay=0.20)

    t0 = time.perf_counter()
    comp_res = client.post(
        "/pipelines/compare",
        json={
            "query": "What is the liability clause in the MSA?",
            "pipeline_a_id": p_a_id,
            "pipeline_b_id": p_b_id,
        },
    )
    total_duration = time.perf_counter() - t0

    assert comp_res.status_code == 200
    comp_data = comp_res.json()

    assert comp_data["query"] == "What is the liability clause in the MSA?"
    assert comp_data["pipeline_a"]["pipeline_id"] == p_a_id
    assert comp_data["pipeline_b"]["pipeline_id"] == p_b_id
    assert "generation" in comp_data["pipeline_a"]
    assert "generation" in comp_data["pipeline_b"]
    assert comp_data["pipeline_a"]["total_latency_ms"] > 0
    assert comp_data["pipeline_b"]["total_latency_ms"] > 0

    # Parallel Execution Timing Proof:
    # If sequential: 0.20s + 0.20s = 0.40s+
    # If concurrent: approx max(0.20s, 0.20s) = ~0.20s - 0.32s
    # Verify total_duration is well under sequential threshold (< 0.38s)
    assert total_duration < 0.38, f"Expected parallel execution < 0.38s, but took {total_duration:.4f}s"


# ============================================================================
# 5. Statistical Percentile & Analytics Tests
# ============================================================================


def test_percentile_calculation_with_known_dataset():
    """Verify p50, p95, p99 against standard mathematical reference values."""
    # Dataset of 10 values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)
    p99 = calculate_percentile(latencies, 99.0)

    # Standard linear interpolation:
    # p50 = 55.0
    # p95 = 90 + 0.55*(100-90) = 95.5
    # p99 = 90 + 0.91*(100-90) = 99.1
    assert p50 == 55.0
    assert p95 == 95.5
    assert p99 == 99.1

    # Single value edge cases
    assert calculate_percentile([42.0], 95.0) == 42.0
    assert calculate_percentile([], 95.0) == 0.0


def test_token_cost_calculation():
    """Verify cost calculation across models and fallback."""
    # gpt-4o-mini: input 0.15/1M, output 0.60/1M
    cost = calculate_run_cost(input_tokens=1_000_000, output_tokens=1_000_000, model_used="gpt-4o-mini")
    assert round(cost, 4) == 0.75

    # gpt-4o: input 2.50/1M, output 10.00/1M
    cost_4o = calculate_run_cost(input_tokens=1_000_000, output_tokens=1_000_000, model_used="gpt-4o")
    assert round(cost_4o, 4) == 12.50


def test_pipeline_analytics_endpoint(task8_client_and_db):
    """Test /pipelines/{id}/analytics returns statistical latency, costs, and 30-day activity."""
    client, db, _ = task8_client_and_db

    # Create pipeline
    cfg = get_valid_pipeline_config_dict("analytics_test_pipeline")
    p_res = client.post("/pipelines", json=cfg)
    p_id = p_res.json()["id"]

    # Seed runs with known latencies: 50, 100, 150, 200
    for lat in [50, 100, 150, 200]:
        db.pipeline_runs.append({
            "id": uuid4(),
            "pipeline_id": UUID(p_id),
            "pipeline_version": 1,
            "query": f"Query with latency {lat}",
            "latency_ms": lat,
            "input_tokens": 1000,
            "output_tokens": 500,
            "model_used": "gpt-4o-mini",
            "status": "complete",
            "created_at": datetime.now(timezone.utc),
            "faithfulness": 0.95,
            "answer_relevance": 0.90,
            "context_precision": 0.85,
            "context_recall": 0.88,
            "overall_score": 0.91,
        })

    analytics_res = client.get(f"/pipelines/{p_id}/analytics")
    assert analytics_res.status_code == 200
    data = analytics_res.json()

    assert "retrieval_latency_p50" in data
    assert "retrieval_latency_p95" in data
    assert "retrieval_latency_p99" in data
    assert data["total_runs"] == 4
    assert data["avg_generation_latency_ms"] == 125.0
    assert data["avg_latency_ms"] == 125.0
    assert data["cost_per_query"] > 0
    assert data["avg_evaluation_scores"]["faithfulness"] == 0.95
    assert len(data["queries_per_day"]) >= 1


def test_pipeline_analytics_empty_pipeline(task8_client_and_db):
    """Test analytics for a pipeline with zero runs returns zeros and nulls cleanly."""
    client, db, _ = task8_client_and_db

    cfg = get_valid_pipeline_config_dict("empty_analytics_pipeline")
    p_res = client.post("/pipelines", json=cfg)
    p_id = p_res.json()["id"]

    analytics_res = client.get(f"/pipelines/{p_id}/analytics")
    assert analytics_res.status_code == 200
    data = analytics_res.json()

    assert data["total_runs"] == 0
    assert data["retrieval_latency_p50"] == 0.0
    assert data["latency_p50_ms"] == 0.0
    assert data["avg_latency_ms"] == 0.0
    assert data["cost_per_query"] == 0.0
    assert data["total_cost_usd"] == 0.0
    assert data["avg_evaluation_scores"]["faithfulness"] is None
    assert data["avg_evaluation_scores"]["overall_score"] is None
    assert data["queries_per_day"] == []
    assert data["recent_failures"] == []


def test_pipeline_analytics_with_failures(task8_client_and_db):
    """Test analytics returns recent failures list correctly."""
    client, db, _ = task8_client_and_db

    cfg = get_valid_pipeline_config_dict("failure_analytics_pipeline")
    p_res = client.post("/pipelines", json=cfg)
    p_id = p_res.json()["id"]

    db.pipeline_runs.append({
        "id": uuid4(),
        "pipeline_id": UUID(p_id),
        "pipeline_version": 1,
        "query": "What is the capital of Mars?",
        "latency_ms": 50,
        "input_tokens": 10,
        "output_tokens": 0,
        "model_used": "gpt-4o-mini",
        "status": "failed",
        "generation": "Rate limit exceeded on provider API",
        "created_at": datetime.now(timezone.utc),
    })

    analytics_res = client.get(f"/pipelines/{p_id}/analytics")
    assert analytics_res.status_code == 200
    data = analytics_res.json()

    assert data["total_runs"] == 1
    assert len(data["recent_failures"]) == 1
    assert data["recent_failures"][0]["query"] == "What is the capital of Mars?"
    assert "Rate limit exceeded" in data["recent_failures"][0]["error_message"]


def test_pipeline_summary_metrics_isolation(task8_client_and_db):
    """Test that pipeline summary metrics are strictly isolated and reflect genuine database records."""
    client, db, _ = task8_client_and_db

    # Pipeline A: 7 runs
    pA_res = client.post("/pipelines", json=get_valid_pipeline_config_dict("Pipeline_Alpha"))
    pA_id = UUID(pA_res.json()["id"])
    for i in range(7):
        r_id = uuid4()
        db.pipeline_runs.append({
            "id": r_id,
            "pipeline_id": pA_id,
            "pipeline_version": 1,
            "query": f"Alpha query {i}",
            "latency_ms": 100 + i * 10,
            "input_tokens": 50,
            "output_tokens": 20,
            "model_used": "gpt-4o-mini",
            "status": "complete",
            "generation": "Alpha response",
            "created_at": datetime.now(timezone.utc),
        })
        db.evaluations.append({
            "id": uuid4(),
            "run_id": r_id,
            "pipeline_id": pA_id,
            "faithfulness": 0.95,
            "answer_relevance": 0.90,
            "context_precision": 0.88,
            "context_recall": 0.92,
            "overall_score": 0.93,
        })

    # Pipeline B: 3 runs
    pB_res = client.post("/pipelines", json=get_valid_pipeline_config_dict("Pipeline_Beta"))
    pB_id = UUID(pB_res.json()["id"])
    for i in range(3):
        r_id = uuid4()
        db.pipeline_runs.append({
            "id": r_id,
            "pipeline_id": pB_id,
            "pipeline_version": 1,
            "query": f"Beta query {i}",
            "latency_ms": 200 + i * 20,
            "input_tokens": 80,
            "output_tokens": 40,
            "model_used": "gpt-4o-mini",
            "status": "complete",
            "generation": "Beta response",
            "created_at": datetime.now(timezone.utc),
        })

    # Pipeline C: 0 runs
    pC_res = client.post("/pipelines", json=get_valid_pipeline_config_dict("Pipeline_Zero"))
    pC_id = UUID(pC_res.json()["id"])

    # Fetch all pipelines
    list_res = client.get("/pipelines")
    assert list_res.status_code == 200
    pipelines = {p["id"]: p for p in list_res.json()}

    # Verify Pipeline Alpha (A)
    summary_A = pipelines[str(pA_id)]["metrics_summary"]
    assert summary_A is not None
    assert summary_A["has_data"] is True
    assert summary_A["queries_7d"] == 7
    assert summary_A["quality_score"] == 0.93
    assert summary_A["faithfulness"] == 0.95
    assert summary_A["latency_p50_ms"] == 130.0
    assert len(summary_A["trend_7d"]) == 7

    # Verify Pipeline Beta (B)
    summary_B = pipelines[str(pB_id)]["metrics_summary"]
    assert summary_B is not None
    assert summary_B["has_data"] is True
    assert summary_B["queries_7d"] == 3
    assert summary_B["latency_p50_ms"] == 220.0

    # Verify Pipeline Zero (C) - Zero runs empty state
    summary_C = pipelines[str(pC_id)]["metrics_summary"]
    assert summary_C is not None
    assert summary_C["has_data"] is False
    assert summary_C["queries_7d"] == 0
    assert summary_C["quality_score"] is None
    assert summary_C["faithfulness"] is None
    assert summary_C["latency_p50_ms"] is None
    assert all(point["query_count"] == 0 for point in summary_C["trend_7d"])

    # Verify isolation: Alpha != Beta != Zero
    assert summary_A["queries_7d"] != summary_B["queries_7d"]
    assert summary_B["queries_7d"] != summary_C["queries_7d"]
    assert summary_A["latency_p50_ms"] != summary_B["latency_p50_ms"]

