from contextlib import asynccontextmanager

from fastapi import FastAPI
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

from api.ingest import router as ingest_router
from config import settings
from db.health import check_mlflow, check_postgres, check_redis
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

    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
        )

    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=settings.anthropic_api_key,
        )

    # -----------------------------
    # NeuroFlow client
    # -----------------------------

    app.state.neuroflow_client = NeuroFlowClient(
        redis=app.state.redis,
        providers=providers,
    )

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

# Automatically create traces for incoming HTTP requests.
app.add_middleware(OpenTelemetryMiddleware)

# Expose Prometheus-compatible metrics at /metrics.
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    """Check the health of NeuroFlow dependencies."""

    postgres_ok = await check_postgres(app.state.db_pool)
    redis_ok = await check_redis()
    mlflow_ok = await check_mlflow()

    all_healthy = postgres_ok and redis_ok and mlflow_ok

    return {
        "status": "ok" if all_healthy else "degraded",
        "checks": {
            "postgres": postgres_ok,
            "redis": redis_ok,
            "mlflow": mlflow_ok,
        },
    }


@app.get("/")
async def root():
    return {"message": "NeuroFlow API"}