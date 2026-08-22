import json
import logging
import re
from typing import Any

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow.evaluation.faithfulness")

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
    "into", "over", "after", "it", "its", "this", "that", "these", "those",
    "and", "or", "but", "if", "while", "as", "than", "so", "what", "which",
    "who", "whom", "where", "when", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "than", "too", "very", "can", "will",
    "just", "should", "now", "also", "did", "does", "do",
}

CLAIM_EXTRACTION_PROMPT = """You are an evaluation assistant. Break down the following text into a list of independent, atomic factual statements/claims.
Return ONLY a valid JSON array of strings, without explanations or markdown formatting.

Example input: "The company was founded in 2012 by Jane Doe in London."
Example output:
[
  "The company was founded in 2012.",
  "The company was founded by Jane Doe.",
  "The company was founded in London."
]
"""

VERIFICATION_PROMPT = """You are a factual verification judge.
Given the provided reference context, determine whether the following claim is supported by the context.
Respond with exactly one word from: ["yes", "no", "partial"].

- "yes": The claim is directly and fully supported by the reference context.
- "partial": The claim is partially supported, or contains minor unsupported details alongside supported facts.
- "no": The claim is contradictory, unsupported, or absent from the reference context.

Respond with ONLY "yes", "no", or "partial"."""


def _evaluate_faithfulness_heuristic(clean_answer: str, clean_context: str) -> float:
    """Evaluate factual alignment and faithfulness of answer assertions against reference context."""
    # Split into independent sentences / clauses
    clauses = [c.strip() for c in re.split(r"[.!?]\s+|,\s+and\s+|,\s+while\s+|,\s+with\s+", clean_answer) if c.strip()]
    if not clauses:
        return 0.0

    context_lower = clean_context.lower()
    total_score = 0.0

    for clause in clauses:
        clause_lower = clause.lower()
        words = re.findall(r"\b[a-z0-9_-]+\b", clause_lower)
        content_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

        if not content_words:
            total_score += 1.0
            continue

        # Extract 2-grams / phrases to test relation binding
        bigrams = [f"{content_words[i]} {content_words[i+1]}" for i in range(len(content_words)-1)]
        matched_bigrams = [bg for bg in bigrams if bg in context_lower]

        # Extract numerical / year / metric entities
        numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", clause_lower)
        has_num_mismatch = any(num not in context_lower for num in numbers)

        # Extract proper nouns (capitalized words in original clause)
        proper_nouns = [w for w in re.findall(r"\b[A-Z][a-z]+\b", clause) if w.lower() not in STOP_WORDS]
        missing_proper_nouns = [pn for pn in proper_nouns if pn.lower() not in context_lower]

        matched_words = [w for w in content_words if w in context_lower]
        word_ratio = len(matched_words) / len(content_words)
        bigram_ratio = len(matched_bigrams) / len(bigrams) if bigrams else word_ratio

        if has_num_mismatch or len(missing_proper_nouns) >= 2 or word_ratio < 0.40:
            # Fact is contradictory or hallucinated
            total_score += 0.0
        elif len(missing_proper_nouns) == 1 or (0.40 <= word_ratio < 0.75) or (bigrams and bigram_ratio < 0.35):
            # Partially supported with ungrounded details
            total_score += 0.5
        elif word_ratio >= 0.75:
            # High lexical & phrase alignment with context
            total_score += 1.0
        else:
            total_score += 0.0

    final_score = total_score / len(clauses)
    return max(0.0, min(1.0, round(final_score, 4)))


async def evaluate_faithfulness(
    query: str,
    answer: str,
    context: str,
    client: Any = None,
) -> float:
    """Evaluate faithfulness: fraction of factual claims in answer supported by the context."""
    clean_answer = (answer or "").strip()
    clean_context = (context or "").strip()

    if not clean_answer:
        return 0.0

    if not clean_context:
        # If context is empty but answer contains content, faithfulness is 0.0
        return 0.0

    if not client:
        return _evaluate_faithfulness_heuristic(clean_answer, clean_context)

    # 1. Extract claims using LLM
    extract_messages = [
        ChatMessage(role="system", content=CLAIM_EXTRACTION_PROMPT),
        ChatMessage(role="user", content=f"Text:\n{clean_answer}"),
    ]

    claims: list[str] = []
    try:
        criteria = RoutingCriteria(task_type="evaluation")
        extract_res = await client.chat(messages=extract_messages, routing_criteria=criteria)
        raw_text = extract_res.content.strip()

        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            claims = [str(c).strip() for c in parsed if str(c).strip()]
    except Exception as exc:
        logger.warning("Claim extraction failed: %s. Falling back to heuristic.", exc)
        return _evaluate_faithfulness_heuristic(clean_answer, clean_context)

    if not claims:
        return 1.0

    # 2. Verify each claim against context
    total_score = 0.0
    for claim in claims:
        user_prompt = f"Reference Context:\n{clean_context}\n\nClaim:\n{claim}"
        verify_messages = [
            ChatMessage(role="system", content=VERIFICATION_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        try:
            criteria = RoutingCriteria(task_type="evaluation")
            verify_res = await client.chat(messages=verify_messages, routing_criteria=criteria)
            verdict = verify_res.content.strip().lower()

            if "yes" in verdict:
                total_score += 1.0
            elif "partial" in verdict:
                total_score += 0.5
            else:
                total_score += 0.0
        except Exception as exc:
            logger.warning("Claim verification failed for claim '%s': %s", claim, exc)
            total_score += _evaluate_faithfulness_heuristic(claim, clean_context)

    final_score = total_score / len(claims)
    return max(0.0, min(1.0, round(final_score, 4)))
