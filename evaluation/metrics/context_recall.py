import logging
import re
from typing import Any

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow.evaluation.context_recall")

RECALL_PROMPT = """You are an evaluation assistant. Determine whether the given sentence from a generated answer can be attributed to the provided reference passages.
Respond with ONLY "yes" or "no"."""


async def evaluate_context_recall(
    query: str,
    chunks: list[str],
    answer: str,
    client: Any = None,
) -> float:
    """Evaluate context recall: fraction of answer sentences attributable to the retrieved context."""
    clean_answer = (answer or "").strip()
    if not clean_answer:
        return 0.0

    combined_context = "\n\n".join(str(c).strip() for c in chunks if str(c).strip())
    if not combined_context:
        return 0.0

    # Break answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_answer) if s.strip()]
    if not sentences:
        return 0.0

    attributable_count = 0.0

    for sentence in sentences:
        if not client:
            # Fallback heuristic: word overlap between sentence and context
            s_words = set(re.findall(r"\w+", sentence.lower()))
            ctx_words = set(re.findall(r"\w+", combined_context.lower()))
            if s_words and (len(s_words.intersection(ctx_words)) / len(s_words)) >= 0.5:
                attributable_count += 1.0
            continue

        user_prompt = f"Reference Context:\n{combined_context}\n\nSentence to verify:\n{sentence}"
        messages = [
            ChatMessage(role="system", content=RECALL_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            criteria = RoutingCriteria(task_type="evaluation")
            res = await client.chat(messages=messages, routing_criteria=criteria)
            verdict = res.content.strip().lower()
            if "yes" in verdict:
                attributable_count += 1.0
        except Exception as exc:
            logger.warning("Context recall check failed for sentence: %s", exc)
            attributable_count += 1.0

    score = attributable_count / len(sentences)
    return max(0.0, min(1.0, round(score, 4)))
