from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from redis.asyncio import Redis
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.openrouter_provider import OpenRouterProvider

from api.ingest import router as ingest_router
from api.query import router as query_router
from api.runs import router as runs_router
from api.pipelines import router as pipelines_router
from api.compare import router as compare_router
from api.finetune import router as finetune_router
from config import settings
from db.health import check_mlflow, check_postgres, check_redis, perform_full_health_check
from db.migrations import check_schema
from db.pool import create_pool
from providers.client import NeuroFlowClient, set_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown resources."""

    # -----------------------------
    # PostgreSQL
    # -----------------------------

    app.state.db_pool = await create_pool()

    schema_ready = await check_schema(app.state.db_pool)

    if not schema_ready:
        await app.state.db_pool.close()
        raise RuntimeError(
            "NeuroFlow database schema is not initialized"
        )

    # -----------------------------
    # Redis
    # -----------------------------

    app.state.redis = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        decode_responses=True,
    )

    # -----------------------------
    # LLM providers
    # -----------------------------

    providers = {}

    if settings.openrouter_api_key:
        providers["openrouter"] = OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            embedding_model=settings.openrouter_embedding_model,
            base_url=settings.openrouter_base_url,
        )

    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
        )

    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=settings.anthropic_api_key,
        )

    # Register default models in Redis for ModelRouter
    router_models = []
    if "openrouter" in providers:
        router_models.append({
            "model": settings.openrouter_llm_model,
            "provider": "openrouter",
            "is_judge": True,
            "is_fine_tuned": False,
            "supports_vision": False,
            "context_window": 128_000,
            "estimated_latency_ms": 250,
            "estimated_cost_per_call": 0.0,
            "task_types": ["generation", "query_processing", "reranking", "evaluation"],
        })
    if "openai" in providers:
        router_models.append({
            "model": "gpt-4o-mini",
            "provider": "openai",
            "is_judge": True,
            "is_fine_tuned": False,
            "supports_vision": False,
            "context_window": 128_000,
            "estimated_latency_ms": 200,
            "estimated_cost_per_call": 0.0001,
            "task_types": ["generation", "query_processing", "reranking", "evaluation"],
        })
    if "anthropic" in providers:
        router_models.append({
            "model": "claude-3-5-sonnet",
            "provider": "anthropic",
            "is_judge": True,
            "is_fine_tuned": False,
            "supports_vision": True,
            "context_window": 200_000,
            "estimated_latency_ms": 300,
            "estimated_cost_per_call": 0.001,
            "task_types": ["generation", "query_processing", "evaluation"],
        })

    if router_models:
        import json
        await app.state.redis.set("router:models", json.dumps(router_models))

    # -----------------------------
    # NeuroFlow client
    # -----------------------------

    app.state.neuroflow_client = NeuroFlowClient(
        redis=app.state.redis,
        providers=providers,
    )
    set_client(app.state.neuroflow_client)

    # -----------------------------
    # ARQ Redis queue pool
    # -----------------------------
    try:
        from arq import create_pool as create_arq_pool
        from arq.connections import RedisSettings

        app.state.arq_redis = await create_arq_pool(
            RedisSettings(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
            )
        )
    except Exception:
        app.state.arq_redis = None

    yield

    # -----------------------------
    # Shutdown
    # -----------------------------

    for provider in providers.values():
        close_method = getattr(provider.client, "close", None)

        if close_method:
            await close_method()

    if getattr(app.state, "arq_redis", None):
        await app.state.arq_redis.close()

    await app.state.redis.aclose()
    await app.state.db_pool.close()


# -----------------------------
# OpenTelemetry configuration
# -----------------------------

resource = Resource.create(
    {
        "service.name": "neuroflow-api",
    }
)

tracer_provider = TracerProvider(resource=resource)

span_exporter = OTLPSpanExporter(
    endpoint=settings.otel_exporter_endpoint,
)

tracer_provider.add_span_processor(
    BatchSpanProcessor(span_exporter)
)

trace.set_tracer_provider(tracer_provider)


# -----------------------------
# FastAPI application
# -----------------------------

app = FastAPI(
    title="NeuroFlow API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(runs_router)
app.include_router(pipelines_router)
app.include_router(compare_router)
app.include_router(finetune_router)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically create traces for incoming HTTP requests.
app.add_middleware(OpenTelemetryMiddleware)

# Expose Prometheus-compatible metrics at /metrics.
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health(request: Request):
    """Check the health and resilience status of NeuroFlow dependencies and circuit breakers."""
    db_pool = getattr(app.state, "db_pool", None)
    redis_client = getattr(app.state, "redis", None) or getattr(app.state, "arq_redis", None)

    return await perform_full_health_check(
        db_pool=db_pool,
        redis_client=redis_client,
    )


@app.get("/")
async def root():
    return {"message": "NeuroFlow API"}