from collections import defaultdict
from pipelines.retrieval.models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Combine multiple retrieval candidate lists using Reciprocal Rank Fusion (RRF).

    Formula: score(d) = sum(1 / (k + rank_m(d))) for each list m where document d appears.
    Rank is 1-indexed.
    """
    if not result_lists:
        return []

    # Map chunk_id to accumulated RRF score and chunk representation
    scores: dict[str, float] = defaultdict(float)
    chunk_store: dict[str, RetrievalResult] = {}
    sources_seen: dict[str, set[str]] = defaultdict(set)

    for result_list in result_lists:
        if not result_list:
            continue
        for rank_idx, item in enumerate(result_list, start=1):
            key = str(item.chunk_id)
            scores[key] += 1.0 / (k + rank_idx)
            sources_seen[key].add(item.source)

            # Store the highest quality metadata/content instance
            if key not in chunk_store:
                chunk_store[key] = item
            else:
                # Merge metadata
                if item.metadata:
                    chunk_store[key].metadata.update(item.metadata)

    # Sort all candidates by descending fused RRF score
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    fused_results: list[RetrievalResult] = []
    for rank, (chunk_id_str, fused_score) in enumerate(sorted_items, start=1):
        original = chunk_store[chunk_id_str]
        merged_source = "+".join(sorted(sources_seen[chunk_id_str])) if len(sources_seen[chunk_id_str]) > 1 else original.source
        fused_results.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                content=original.content,
                score=fused_score,
                rank=rank,
                source="fused",
                filename=original.filename,
                page_number=original.page_number,
                metadata={**original.metadata, "fused_from": merged_source},
            )
        )

    return fused_results
