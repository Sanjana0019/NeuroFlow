import asyncio
from backend.config import settings
from backend.db.pool import create_pool
from backend.db.documents import DocumentRepository
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.providers.client import NeuroFlowClient
from pipelines.ingestion.chunker import Chunker
from pipelines.ingestion.embedder import ChunkEmbedder
from pipelines.ingestion.dispatcher import IngestionDispatcher
from redis.asyncio import Redis

async def test_ingest():
    print("Testing document ingestion...")
    pool = await create_pool()
    repo = DocumentRepository(pool)
    redis = Redis(host=settings.redis_host, port=settings.redis_port, password=settings.redis_password, decode_responses=True)
    
    provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    client = NeuroFlowClient(redis=redis, providers={"openrouter": provider})

    raw_text = """
    NeuroFlow is an enterprise-grade Retrieval-Augmented Generation (RAG) system.
    It integrates dense vector retrieval with NVIDIA Nemotron 3 Embed (2048 dimensions),
    BM25 sparse keyword search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
    """
    
    dispatcher = IngestionDispatcher()
    parsed_doc = await dispatcher.dispatch(raw_text.encode("utf-8"), "test_architecture.txt", "text/plain")
    print(f"Parsed text length: {len(parsed_doc.text)}")

    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(parsed_doc)
    print(f"Created {len(chunks)} chunks.")

    embedder = ChunkEmbedder(client=client, expected_dimension=2048)
    embedded_chunks = await embedder.embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks with 2048-dim vectors!")

    doc_id, count = await repo.save_document_with_chunks(
        filename="test_architecture.txt",
        source_type="text/plain",
        chunks=embedded_chunks,
        status="completed",
    )
    print(f"Document ingestion SUCCESS! Document {doc_id} saved with {count} chunks in status 'completed'.")

    await pool.close()
    await redis.close()

if __name__ == "__main__":
    asyncio.run(test_ingest())
