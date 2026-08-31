import asyncio
import json
from uuid import uuid4
import asyncpg
from backend.config import settings

async def seed_data():
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )

    try:
        # Check pipelines
        count = await conn.fetchval("SELECT COUNT(*) FROM pipelines")
        if count == 0:
            pipe1_id = uuid4()
            pipe1_config = {
                "name": "Production-Hybrid-RAG",
                "description": "Default production pipeline with hybrid dense/sparse retrieval and cross-encoder reranking.",
                "retrieval": {
                    "dense_k": 20,
                    "sparse_k": 15,
                    "top_k_after_rerank": 5,
                    "query_expansion": True,
                    "similarity_threshold": 0.7,
                    "rrf_k": 60,
                },
                "generation": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_context_tokens": 4000,
                    "system_prompt": "You are a precise enterprise research assistant. Cite sources using [Source N].",
                },
                "evaluation": {
                    "auto_evaluate": True,
                    "judge_model": "gpt-4o-mini",
                },
                "rate_limit_rpm": 60,
            }

            await conn.execute(
                """
                INSERT INTO pipelines (id, name, description, version, status, config, created_at, updated_at)
                VALUES ($1, $2, $3, 1, 'active', $4, NOW(), NOW())
                """,
                pipe1_id,
                pipe1_config["name"],
                pipe1_config["description"],
                json.dumps(pipe1_config),
            )
            await conn.execute(
                """
                INSERT INTO pipeline_versions (pipeline_id, version, config, created_at)
                VALUES ($1, 1, $2, NOW())
                """,
                pipe1_id,
                json.dumps(pipe1_config),
            )

            pipe2_id = uuid4()
            pipe2_config = {
                "name": "Fast-Dense-Search",
                "description": "Low-latency semantic dense vector retrieval pipeline for high-throughput queries.",
                "retrieval": {
                    "dense_k": 10,
                    "sparse_k": 0,
                    "top_k_after_rerank": 3,
                    "query_expansion": False,
                    "similarity_threshold": 0.8,
                },
                "generation": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_context_tokens": 2000,
                    "system_prompt": "Provide concise direct answers based on [Source N].",
                },
                "evaluation": {
                    "auto_evaluate": True,
                    "judge_model": "gpt-4o-mini",
                },
                "rate_limit_rpm": 120,
            }

            await conn.execute(
                """
                INSERT INTO pipelines (id, name, description, version, status, config, created_at, updated_at)
                VALUES ($1, $2, $3, 1, 'active', $4, NOW(), NOW())
                """,
                pipe2_id,
                pipe2_config["name"],
                pipe2_config["description"],
                json.dumps(pipe2_config),
            )
            await conn.execute(
                """
                INSERT INTO pipeline_versions (pipeline_id, version, config, created_at)
                VALUES ($1, 1, $2, NOW())
                """,
                pipe2_id,
                json.dumps(pipe2_config),
            )
            print("Successfully seeded 2 default pipelines.")
        else:
            print(f"Pipelines table already has {count} entries.")

        # Check sample documents
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        if doc_count == 0:
            doc_id = uuid4()
            await conn.execute(
                """
                INSERT INTO documents (id, filename, source_type, content_hash, status, chunk_count, metadata, created_at)
                VALUES ($1, 'NeuroFlow_Architecture_Overview.pdf', 'pdf', 'hash123', 'completed', 3, '{"author":"NeuroFlow Team"}', NOW())
                """,
                doc_id,
            )

            # Insert sample chunks with embeddings
            sample_chunks = [
                (uuid4(), doc_id, 0, "NeuroFlow implements a distributed hybrid retrieval pipeline combining pgvector embeddings, sparse BM25 indexing, and metadata filters with reciprocal rank fusion (RRF).", 28),
                (uuid4(), doc_id, 1, "Cross-encoder reranking refines candidate passages to maximize context precision before prompt assembly and generation.", 22),
                (uuid4(), doc_id, 2, "Asynchronous evaluation runs automated LLM Judge scoring across Faithfulness, Answer Relevance, Context Precision, and Context Recall metrics.", 25),
            ]

            for chunk_id, d_id, c_idx, content, tokens in sample_chunks:
                await conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, chunk_index, content, token_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, '{}')
                    """,
                    chunk_id,
                    d_id,
                    c_idx,
                    content,
                    tokens,
                )

            # Seed a sample evaluation run
            run_id = uuid4()
            pipe_id = await conn.fetchval("SELECT id FROM pipelines LIMIT 1")
            await conn.execute(
                """
                INSERT INTO pipeline_runs (id, pipeline_id, pipeline_version, query, generation, retrieved_chunk_ids, latency_ms, input_tokens, output_tokens, model_used, status, created_at)
                VALUES ($1, $2, 1, 'How does NeuroFlow perform hybrid retrieval?', 'Based on [Source 1], NeuroFlow combines dense pgvector embeddings with sparse BM25 search using Reciprocal Rank Fusion (RRF).', $3, 340, 120, 45, 'gpt-4o-mini', 'complete', NOW())
                """,
                run_id,
                pipe_id,
                [sample_chunks[0][0], sample_chunks[1][0]],
            )

            await conn.execute(
                """
                INSERT INTO evaluations (run_id, faithfulness, answer_relevance, context_precision, context_recall, overall_score, judge_model, evaluated_at)
                VALUES ($1, 0.96, 0.94, 0.88, 0.85, 0.92, 'gpt-4o-mini', NOW())
                """,
                run_id,
            )
            print("Successfully seeded sample document, chunks, and evaluation run.")
        else:
            print(f"Documents table already has {doc_count} entries.")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_data())
