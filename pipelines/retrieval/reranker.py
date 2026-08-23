import asyncio
import logging
import re

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria
from backend.resilience.timeout_manager import execute_with_timeout
from pipelines.retrieval.models import RetrievalResult

logger = logging.getLogger("neuroflow.retrieval.reranker")

RERANK_SYSTEM_PROMPT = """You are a relevance scoring judge for an information retrieval system.
Given a user query and a retrieved document passage, score the passage relevance to the query on a strict scale from 0.0 to 10.0:
- 10.0: Perfectly relevant and directly answers the query completely.
- 7.0-9.9: Highly relevant, containing key facts directly addressing the query.
- 4.0-6.9: Partially relevant or related topic context without directly answering.
- 1.0-3.9: Tangentially mentions words from the query but irrelevant.
- 0.0: Completely irrelevant.

Respond with ONLY the numeric score (e.g. "8.5" or "9.0"). Do not provide explanations or words."""


class Reranker:
    """Reranks candidate retrieval chunks using LLM cross-encoder relevance scoring."""

    def __init__(self, client=None, concurrency: int = 10, max_candidates: int = 40):
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_candidates = max_candidates

    async def _score_pair(self, query: str, candidate: RetrievalResult) -> float:
        """Score a single (query, chunk.content) candidate using LLM or heuristic."""
        if not self.client:
            # Fallback scoring using token overlap and position
            query_words = set(re.findall(r"\w+", query.lower()))
            content_words = set(re.findall(r"\w+", candidate.content.lower()))
            if not query_words or not content_words:
                return candidate.score
            overlap = len(query_words.intersection(content_words)) / len(query_words)
            return round(overlap * 10.0, 2)

        async with self.semaphore:
            user_prompt = f"Query: {query}\n\nPassage:\n{candidate.content}"
            messages = [
                ChatMessage(role="system", content=RERANK_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ]

            try:
                criteria = RoutingCriteria(task_type="reranking")
                result = await self.client.chat(messages=messages, routing_criteria=criteria)
                raw_score = result.content.strip()

                # Extract numeric score (0 to 10)
                match = re.search(r"(\d+(?:\.\d+)?)", raw_score)
                if match:
                    val = float(match.group(1))
                    return max(0.0, min(10.0, val))
                return candidate.score * 10.0
            except Exception as exc:
                logger.warning("Reranking failed for candidate %s: %s", candidate.chunk_id, exc)
                return candidate.score * 10.0

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates in parallel and return the top-K highest scoring chunks."""
        if not candidates:
            return []

        # Restrict to max_candidates for reranking
        selected_candidates = candidates[: self.max_candidates]

        # Score candidates in parallel with TimeoutManager
        tasks = [self._score_pair(query, candidate) for candidate in selected_candidates]
        async def _gather_scores():
            return await asyncio.gather(*tasks, return_exceptions=True)

        redis_client = getattr(self.client, "redis", None) if self.client else None
        scores = await execute_with_timeout("reranking", _gather_scores(), redis=redis_client)

        scored_candidates: list[tuple[float, RetrievalResult]] = []
        for idx, candidate in enumerate(selected_candidates):
            score = scores[idx]
            if isinstance(score, Exception) or score is None:
                final_score = candidate.score * 10.0
            else:
                final_score = float(score)

            scored_candidates.append((final_score, candidate))

        # Sort descending by relevance score
        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        reranked_results: list[RetrievalResult] = []
        for rank, (score, orig) in enumerate(scored_candidates, start=1):
            reranked_results.append(
                RetrievalResult(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    content=orig.content,
                    score=score,
                    rank=rank,
                    source="reranked",
                    filename=orig.filename,
                    page_number=orig.page_number,
                    metadata={**orig.metadata, "rerank_score": score},
                )
            )

        if top_k is not None:
            return reranked_results[:top_k]
        return reranked_results
