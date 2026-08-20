from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from api.ingest import router as ingest_router

from config import settings
from db.health import check_mlflow, check_postgres, check_redis
from db.migrations import check_schema
from db.pool import create_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown resources."""
    app.state.db_pool = await create_pool()

    schema_ready = await check_schema(app.state.db_pool)

    if not schema_ready:
        raise RuntimeError("NeuroFlow database schema is not initialized")

    yield

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