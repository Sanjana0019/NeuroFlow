import logging
import re
from typing import Any

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow.evaluation.context_precision")

PRECISION_PROMPT = """You are an evaluation assistant. Determine whether the provided retrieved passage is useful and relevant for answering the user query.
Respond with ONLY "yes" or "no"."""


async def evaluate_context_precision(
    query: str,
    chunks: list[str],
    answer: str,
    client: Any = None,
) -> float:
    """Evaluate context precision by rank-weighting useful retrieved passages: sum(useful[i]*(1/i)) / sum(1/i)."""
    if not chunks:
        return 0.0

    clean_chunks = [str(c).strip() for c in chunks if str(c).strip()]
    if not clean_chunks:
        return 0.0

    useful_indicators: list[float] = []

    for rank_idx, chunk in enumerate(clean_chunks, start=1):
        if not client:
            # Fallback heuristic: word overlap between chunk and (query + answer)
            combined_words = set(re.findall(r"\w+", (query + " " + answer).lower()))
            chunk_words = set(re.findall(r"\w+", chunk.lower()))
            overlap = len(chunk_words.intersection(combined_words))
            useful_indicators.append(1.0 if overlap >= 2 else 0.0)
            continue

        user_prompt = f"Query: {query}\n\nAnswer: {answer}\n\nRetrieved Passage:\n{chunk}"
        messages = [
            ChatMessage(role="system", content=PRECISION_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            criteria = RoutingCriteria(task_type="evaluation")
            res = await client.chat(messages=messages, routing_criteria=criteria)
            verdict = res.content.strip().lower()
            useful_indicators.append(1.0 if "yes" in verdict else 0.0)
        except Exception as exc:
            logger.warning("Context precision check failed for chunk %d: %s", rank_idx, exc)
            useful_indicators.append(1.0)

    # Weighted precision calculation: sum(useful[i]*(1/i)) / sum(1/i)
    weighted_useful = sum(useful_indicators[i] * (1.0 / (i + 1)) for i in range(len(useful_indicators)))
    total_weights = sum(1.0 / (i + 1) for i in range(len(useful_indicators)))

    if total_weights <= 0.0:
        return 0.0

    score = weighted_useful / total_weights
    return max(0.0, min(1.0, round(score, 4)))
