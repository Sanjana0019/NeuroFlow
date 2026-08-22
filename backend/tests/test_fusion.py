from uuid import uuid4

from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.models import RetrievalResult


def make_chunk(chunk_id=None, content="Sample content", score=1.0, rank=1, source="dense"):
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        content=content,
        score=score,
        rank=rank,
        source=source,
        filename="doc.pdf",
    )


def test_rrf_score_calculation_and_boosting():
    """Verify formula score = sum(1 / (k + rank)) and multi-list boosting."""
    c_shared = make_chunk(content="Shared Chunk in both lists")
    c_dense_only = make_chunk(content="Dense only chunk")
    c_sparse_only = make_chunk(content="Sparse only chunk")

    # List 1 (Dense): rank 1 for c_shared, rank 2 for c_dense_only
    dense_list = [
        RetrievalResult(chunk_id=c_shared.chunk_id, document_id=c_shared.document_id, content=c_shared.content, rank=1, source="dense"),
        RetrievalResult(chunk_id=c_dense_only.chunk_id, document_id=c_dense_only.document_id, content=c_dense_only.content, rank=2, source="dense"),
    ]

    # List 2 (Sparse): rank 1 for c_sparse_only, rank 2 for c_shared
    sparse_list = [
        RetrievalResult(chunk_id=c_sparse_only.chunk_id, document_id=c_sparse_only.document_id, content=c_sparse_only.content, rank=1, source="sparse"),
        RetrievalResult(chunk_id=c_shared.chunk_id, document_id=c_shared.document_id, content=c_shared.content, rank=2, source="sparse"),
    ]

    fused = reciprocal_rank_fusion([dense_list, sparse_list], k=60)

    assert len(fused) == 3

    # c_shared was rank 1 in list 1 (1/61) and rank 2 in list 2 (1/62)
    # Expected score: 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
    expected_shared_score = (1.0 / 61.0) + (1.0 / 62.0)
    expected_single_score = 1.0 / 61.0

    top_result = fused[0]
    assert top_result.chunk_id == c_shared.chunk_id
    assert abs(top_result.score - expected_shared_score) < 1e-6
    assert top_result.rank == 1
    assert top_result.source == "fused"

    # Single-list items have score 1/61
    assert abs(fused[1].score - expected_single_score) < 1e-6
    assert abs(fused[2].score - 1.0 / 62.0) < 1e-6


def test_rrf_empty_and_single_list():
    """RRF handles empty lists gracefully."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []

    c1 = make_chunk(content="Only chunk")
    fused = reciprocal_rank_fusion([[c1]], k=60)
    assert len(fused) == 1
    assert fused[0].chunk_id == c1.chunk_id
    assert abs(fused[0].score - (1.0 / 61.0)) < 1e-6
