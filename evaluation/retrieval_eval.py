import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_DNS

from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.models import ProcessedQuery, RetrievalResult
from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import Reranker
from pipelines.retrieval.retriever import Retriever


def compute_semantic_embedding(text: str, dim: int = 1536) -> list[float]:
    """Deterministic simulated embedding projecting text semantics into a 1536-dim vector."""
    clean = text.lower()
    words = re.findall(r"\w+", clean)
    vec = [0.0] * dim

    for w in words:
        # Hash word into multiple dimensions
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        for i in range(16):
            idx = (h + i * 97) % dim
            val = ((h >> (i * 4)) & 0xFF) / 255.0
            vec[idx] += val

    # Normalize L2 norm
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))


class EvalEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [compute_semantic_embedding(t) for t in texts]


class EvalMemoryDB:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.chunks_table: list[dict] = []

        for doc in documents:
            doc_id = uuid5(NAMESPACE_DNS, doc["id"])
            for c in doc["chunks"]:
                chunk_uuid = uuid5(NAMESPACE_DNS, c["id"])
                emb = compute_semantic_embedding(c["content"])
                self.chunks_table.append(
                    {
                        "id": chunk_uuid,
                        "raw_id": c["id"],
                        "document_id": doc_id,
                        "content": c["content"],
                        "chunk_index": 0,
                        "token_count": len(c["content"].split()),
                        "metadata": c.get("metadata", {}),
                        "doc_metadata": doc.get("metadata", {}),
                        "filename": doc["filename"],
                        "embedding": emb,
                    }
                )

    def acquire(self):
        db = self

        class Conn:
            async def fetch(self, query: str, *args):
                normalized = " ".join(query.split())

                if "WHERE (c.metadata @> $2::jsonb" in normalized:
                    # Metadata query
                    filter_json = json.loads(args[1]) if len(args) > 1 else {}
                    k = args[2] if len(args) > 2 else 20
                    q_vec = [float(x) for x in args[0].strip("[]").split(",")] if len(args) > 0 else []

                    matches = []
                    for row in db.chunks_table:
                        # Match metadata filter
                        c_meta = row["metadata"]
                        d_meta = row["doc_metadata"]
                        matches_all = True
                        for k_f, v_f in filter_json.items():
                            if c_meta.get(k_f) != v_f and d_meta.get(k_f) != v_f:
                                matches_all = False
                                break
                        if matches_all:
                            score = cosine_similarity(q_vec, row["embedding"])
                            matches.append({**row, "score": score})

                    matches.sort(key=lambda r: r["score"], reverse=True)
                    return matches[:k]

                elif "to_tsvector" in normalized:
                    # Sparse lexical FTS
                    query_str = args[0].lower()
                    k = args[1] if len(args) > 1 else 20
                    q_words = set(re.findall(r"\w+", query_str))

                    scored = []
                    for row in db.chunks_table:
                        c_words = set(re.findall(r"\w+", row["content"].lower()))
                        overlap = len(q_words.intersection(c_words))
                        if overlap > 0:
                            score = overlap / len(q_words)
                            scored.append({**row, "score": score})

                    scored.sort(key=lambda r: r["score"], reverse=True)
                    return scored[:k]

                else:
                    # Dense vector search
                    k = args[1] if len(args) > 1 else 20
                    q_vec = [float(x) for x in args[0].strip("[]").split(",")] if len(args) > 0 else []

                    scored = []
                    for row in db.chunks_table:
                        score = cosine_similarity(q_vec, row["embedding"])
                        scored.append({**row, "score": score})

                    scored.sort(key=lambda r: r["score"], reverse=True)
                    return scored[:k]

        class AcquireCtx:
            async def __aenter__(self):
                return Conn()
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        return AcquireCtx()


async def run_evaluation():
    dataset_path = Path("evaluation/retrieval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    db = EvalMemoryDB(dataset["documents"])
    embedder = EvalEmbedder()
    retriever = Retriever(db_pool=db, embedder=embedder)
    query_processor = QueryProcessor(client=None)  # Uses heuristic query processing
    reranker = Reranker(client=None)  # Uses relevance scoring
    context_assembler = ContextAssembler(token_budget=4000)

    pipeline = RetrievalPipeline(
        retriever=retriever,
        query_processor=query_processor,
        reranker=reranker,
        context_assembler=context_assembler,
    )

    # Chunk ID mapping
    uuid_to_raw_id = {row["id"]: row["raw_id"] for row in db.chunks_table}

    modes = ["dense_only", "rrf_only", "full"]
    metrics: dict[str, dict[str, Any]] = {}

    for mode in modes:
        hits = 0
        reciprocal_ranks: list[float] = []
        per_query_details: list[dict] = []

        for q_item in dataset["queries"]:
            q_text = q_item["query"]
            expected_ids = set(q_item["expected_chunk_ids"])

            chunks, assembled_ctx, processed_q = await pipeline.run(
                query=q_text,
                mode=mode,  # type: ignore
                top_k=5,
                retrieval_k=20,
            )

            retrieved_raw_ids = [uuid_to_raw_id.get(c.chunk_id, str(c.chunk_id)) for c in chunks]

            # Calculate Hit Rate & Reciprocal Rank
            hit = False
            rr = 0.0
            for rank_idx, r_id in enumerate(retrieved_raw_ids, start=1):
                if r_id in expected_ids:
                    if not hit:
                        hit = True
                        rr = 1.0 / rank_idx
                        break

            if hit:
                hits += 1
            reciprocal_ranks.append(rr)

            per_query_details.append(
                {
                    "query_id": q_item["id"],
                    "query": q_text,
                    "expected": list(expected_ids),
                    "retrieved": retrieved_raw_ids,
                    "hit": hit,
                    "reciprocal_rank": round(rr, 4),
                }
            )

        total_queries = len(dataset["queries"])
        hit_rate = hits / total_queries
        mrr = sum(reciprocal_ranks) / total_queries

        metrics[mode] = {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "total_queries": total_queries,
            "hits": hits,
            "per_query": per_query_details,
        }

    results_output = {
        "benchmark": "NeuroFlow Production Retrieval Evaluation",
        "thresholds": {
            "target_hit_rate": 0.75,
            "target_mrr": 0.55,
        },
        "baseline_dense": {
            "hit_rate": metrics["dense_only"]["hit_rate"],
            "mrr": metrics["dense_only"]["mrr"],
        },
        "rrf_hybrid": {
            "hit_rate": metrics["rrf_only"]["hit_rate"],
            "mrr": metrics["rrf_only"]["mrr"],
        },
        "rrf_with_reranking": {
            "hit_rate": metrics["full"]["hit_rate"],
            "mrr": metrics["full"]["mrr"],
        },
        "reranking_improved_mrr": metrics["full"]["mrr"] >= metrics["rrf_only"]["mrr"],
        "thresholds_met": (
            metrics["full"]["hit_rate"] >= 0.75 and metrics["full"]["mrr"] >= 0.55
        ),
        "detailed_results": metrics,
    }

    results_path = Path("evaluation/retrieval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)

    print("============================================================")
    print("NeuroFlow Retrieval Pipeline Evaluation Results")
    print("============================================================")
    print(f"1. Naive / Dense Baseline : Hit Rate = {metrics['dense_only']['hit_rate']:.4f} | MRR = {metrics['dense_only']['mrr']:.4f}")
    print(f"2. RRF Hybrid Retrieval   : Hit Rate = {metrics['rrf_only']['hit_rate']:.4f} | MRR = {metrics['rrf_only']['mrr']:.4f}")
    print(f"3. RRF + Reranking (Full) : Hit Rate = {metrics['full']['hit_rate']:.4f} | MRR = {metrics['full']['mrr']:.4f}")
    print("------------------------------------------------------------")
    print(f"Target Thresholds : Hit Rate > 0.75 | MRR > 0.55")
    print(f"Thresholds Passed : {results_output['thresholds_met']}")
    print(f"MRR Improvement   : {results_output['reranking_improved_mrr']}")
    print(f"Results Saved To  : {results_path}")
    print("============================================================")

    return results_output


if __name__ == "__main__":
    asyncio.run(run_evaluation())
