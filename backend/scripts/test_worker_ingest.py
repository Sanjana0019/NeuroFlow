import asyncio
import tempfile
import os
from backend.config import settings
from backend.db.pool import create_pool
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.providers.client import NeuroFlowClient
from backend.worker import process_ingestion_job
from redis.asyncio import Redis

async def test_worker_ingest():
    print("Testing worker process_ingestion_job end-to-end...")
    pool = await create_pool()
    redis = Redis(host=settings.redis_host, port=settings.redis_port, password=settings.redis_password, decode_responses=True)
    
    provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    client = NeuroFlowClient(redis=redis, providers={"openrouter": provider})

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("NeuroFlow enterprise RAG system architecture overview.\nIt features hybrid retrieval with dense vector embeddings and BM25 search.\nReciprocal rank fusion ensures top results.")
        temp_path = f.name

    ctx = {
        "db_pool": pool,
        "redis": redis,
        "neuroflow_client": client,
    }

    try:
        res = await process_ingestion_job(
            ctx=ctx,
            source=temp_path,
            filename="architecture_spec.txt",
            source_type="text/plain",
        )
        print("Worker ingestion result:", res)
    finally:
        os.remove(temp_path)
        await pool.close()
        await redis.close()

if __name__ == "__main__":
    asyncio.run(test_worker_ingest())
