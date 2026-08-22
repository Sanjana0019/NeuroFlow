from pathlib import Path
import pytest

from evaluation.retrieval_eval import run_evaluation


@pytest.mark.asyncio
async def test_retrieval_benchmark_evaluation_metrics():
    """Verify that the retrieval pipeline achieves Hit Rate > 0.75 and MRR > 0.55 on the benchmark."""
    results = await run_evaluation()

    assert results["thresholds_met"] is True
    assert results["rrf_with_reranking"]["hit_rate"] >= 0.75
    assert results["rrf_with_reranking"]["mrr"] >= 0.55
    assert results["reranking_improved_mrr"] is True

    # Verify JSON persistence
    results_file = Path("evaluation/retrieval_results.json")
    assert results_file.exists()
