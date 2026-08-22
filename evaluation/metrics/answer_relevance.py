import hashlib
import json
import logging
import math
import re
from typing import Any

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow.evaluation.answer_relevance")

QUESTION_GEN_PROMPT = """You are an evaluation assistant. Based on the provided answer passage, generate 3 to 5 distinct questions that this passage directly and reasonably answers.
Return ONLY a valid JSON array of question strings without extra text or markdown formatting.

Example input: "Paris is the capital and most populous city of France."
Example output:
[
  "What is the capital of France?",
  "Which city is the most populous in France?",
  "What country is Paris the capital of?"
]
"""


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def _heuristic_embedding(text: str, dim: int = 1536) -> list[float]:
    """Deterministic simulated embedding for unit tests and fallback."""
    words = re.findall(r"\w+", text.lower())
    vec = [0.0] * dim
    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        for i in range(16):
            idx = (h + i * 97) % dim
            val = ((h >> (i * 4)) & 0xFF) / 255.0
            vec[idx] += val
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


async def evaluate_answer_relevance(
    query: str,
    answer: str,
    client: Any = None,
    embedder: Any = None,
) -> float:
    """Evaluate answer relevance by generating reverse questions and computing embedding cosine similarity."""
    clean_query = (query or "").strip()
    clean_answer = (answer or "").strip()

    if not clean_query or not clean_answer:
        return 0.0

    generated_questions: list[str] = []

    if client:
        messages = [
            ChatMessage(role="system", content=QUESTION_GEN_PROMPT),
            ChatMessage(role="user", content=f"Answer:\n{clean_answer}"),
        ]
        try:
            criteria = RoutingCriteria(task_type="evaluation")
            res = await client.chat(messages=messages, routing_criteria=criteria)
            raw_text = res.content.strip()

            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                generated_questions = [str(q).strip() for q in parsed if str(q).strip()]
        except Exception as exc:
            logger.warning("Question generation failed: %s. Using heuristic.", exc)

    if not generated_questions:
        generated_questions = [
            f"What does the passage say about {clean_query}?",
            f"Explain {clean_query}",
            f"Key facts regarding {clean_query}",
        ]

    # Resolve embedding service
    embed_service = embedder or getattr(client, "embed", None)
    all_texts = [clean_query] + generated_questions

    if embed_service:
        try:
            if callable(embed_service):
                embeddings = await embed_service(all_texts)
            elif hasattr(embed_service, "embed"):
                embeddings = await embed_service.embed(all_texts)
            else:
                embeddings = [_heuristic_embedding(t) for t in all_texts]
        except Exception as exc:
            logger.warning("Embedding service call failed: %s. Using heuristic embeddings.", exc)
            embeddings = [_heuristic_embedding(t) for t in all_texts]
    else:
        embeddings = [_heuristic_embedding(t) for t in all_texts]

    query_emb = embeddings[0]
    question_embs = embeddings[1:]

    similarities = [_cosine_similarity(query_emb, q_emb) for q_emb in question_embs]
    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

    return max(0.0, min(1.0, round(mean_sim, 4)))
