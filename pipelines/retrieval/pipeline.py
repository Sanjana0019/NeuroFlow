from typing import Any, Literal

from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.fusion import reciprocal_rank_fusion
from pipelines.retrieval.models import AssembledContext, ProcessedQuery, RetrievalResult
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import Reranker
from pipelines.retrieval.retriever import Retriever

PipelineMode = Literal["dense_only", "rrf_only", "full"]


class RetrievalPipeline:
    """Unified orchestration layer for query processing, hybrid retrieval, fusion, reranking, and context assembly."""

    def __init__(
        self,
        retriever: Retriever,
        query_processor: QueryProcessor | None = None,
        reranker: Reranker | None = None,
        context_assembler: ContextAssembler | None = None,
        enable_query_expansion: bool = True,
    ):
        self.retriever = retriever
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker or Reranker()
        self.context_assembler = context_assembler or ContextAssembler()
        self.enable_query_expansion = enable_query_expansion

    async def run(
        self,
        query: str,
        mode: PipelineMode = "full",
        top_k: int = 10,
        retrieval_k: int = 20,
        token_budget: int | None = None,
        override_filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievalResult], AssembledContext, ProcessedQuery]:
        """Execute the retrieval pipeline and return ranked chunks, assembled context, and query metadata."""
        # 1. Query Processing
        processed_query = await self.query_processor.process(query)
        if override_filters:
            processed_query.metadata_filters.update(override_filters)

        if not self.enable_query_expansion:
            processed_query.expanded_queries = []

        # 2. Parallel Retrieval
        if mode == "dense_only":
            chunks = await self.retriever.dense_retrieval(
                query=processed_query.original_query,
                expanded_queries=[],
                k=retrieval_k,
            )
            final_chunks = chunks[:top_k]
        else:
            candidates_by_source = await self.retriever.retrieve_parallel(
                query=processed_query,
                k=retrieval_k,
            )

            # 3. Reciprocal Rank Fusion
            result_lists = [
                candidates_by_source["dense"],
                candidates_by_source["sparse"],
                candidates_by_source["metadata"],
            ]
            fused_candidates = reciprocal_rank_fusion(result_lists, k=60)

            if mode == "rrf_only":
                final_chunks = fused_candidates[:top_k]
            else:
                # 4. Cross-Encoder Reranking
                reranked = await self.reranker.rerank(
                    query=processed_query.original_query,
                    candidates=fused_candidates,
                    top_k=top_k,
                )
                final_chunks = reranked

        # 5. Context Assembly
        assembled_context = self.context_assembler.assemble(
            chunks=final_chunks,
            custom_budget=token_budget,
        )

        return final_chunks, assembled_context, processed_query
