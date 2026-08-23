import json
import logging
from pathlib import Path
import re
from typing import Any
from uuid import UUID

import tiktoken

from evaluation.judge import EvaluationJudge

logger = logging.getLogger("neuroflow.finetuning.extractor")

# PII Regex Patterns
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
CITATION_REGEX = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)


class TrainingDataExtractor:
    """Extracts, validates, filters for PII/quality, and serializes training pairs for fine-tuning."""

    def __init__(
        self,
        token_encoding: str = "cl100k_base",
        training_data_dir: str | Path = "training_data",
    ):
        self.training_data_dir = Path(training_data_dir)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.tokenizer = tiktoken.get_encoding(token_encoding)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def check_pii(self, text: str) -> bool:
        """Return True if text contains email or phone PII."""
        if not text:
            return False
        if EMAIL_REGEX.search(text) or PHONE_REGEX.search(text):
            return True
        return False

    def validate_pair(
        self,
        system_prompt: str | None,
        user_message: str,
        assistant_message: str,
        faithfulness: float | None = None,
    ) -> tuple[bool, str | None]:
        """Validate a single training pair candidate.
        
        Rules:
        - User message and assistant message must not be empty.
        - No email or phone PII in query/user message.
        - Assistant response length must be between 50 and 2000 tokens.
        - Assistant response must contain at least one [Source N] citation.
        - Faithfulness must be > 0.8.
        """
        if not user_message or not assistant_message:
            return False, "Empty user message or assistant message"

        # 1. PII check
        if self.check_pii(user_message):
            return False, "User message contains PII (email or phone)"

        # 2. Token length check (50 - 2000 tokens)
        token_count = self.count_tokens(assistant_message)
        if token_count < 50:
            return False, f"Assistant response too short: {token_count} tokens (min 50)"
        if token_count > 2000:
            return False, f"Assistant response too long: {token_count} tokens (max 2000)"

        # 3. Citation check
        if not CITATION_REGEX.search(assistant_message):
            return False, "Assistant response missing required [Source N] citation"

        # 4. Faithfulness check
        if faithfulness is None or faithfulness <= 0.8:
            return False, f"Faithfulness score {faithfulness} is not > 0.8"

        return True, None

    async def extract_and_validate(
        self,
        db_pool,
        job_id: UUID | str,
        min_quality_score: float = 0.82,
        client: Any = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
        """Extract eligible training pairs, validate, and write validated pairs to JSONL."""
        job_uuid = UUID(str(job_id))
        valid_pairs: list[dict[str, Any]] = []
        rejected_pairs: list[dict[str, Any]] = []

        async with db_pool.acquire() as conn:
            # Query training pairs eligible for extraction
            rows = await conn.fetch(
                """
                SELECT tp.id, tp.run_id, tp.system_prompt, tp.user_message, tp.assistant_message,
                       tp.quality_score, tp.created_at,
                       e.faithfulness, e.user_rating, r.query, r.generation, r.retrieved_chunk_ids
                FROM training_pairs tp
                LEFT JOIN evaluations e ON e.run_id = tp.run_id
                LEFT JOIN pipeline_runs r ON r.id = tp.run_id
                WHERE tp.included_in_job IS NULL
                  AND tp.quality_score >= $1
                  AND (e.user_rating IS NULL OR e.user_rating >= 4)
                ORDER BY tp.created_at ASC
                """,
                min_quality_score,
            )

            for row in rows:
                pair_id = row["id"]
                system_prompt = row["system_prompt"] or "You are a precise research assistant."
                user_msg = row["user_message"]
                assistant_msg = row["assistant_message"]
                faithfulness = row["faithfulness"]

                # If faithfulness is missing, attempt re-evaluation if possible
                if faithfulness is None and client is not None and row["retrieved_chunk_ids"]:
                    try:
                        chunk_rows = await conn.fetch(
                            "SELECT content FROM chunks WHERE id = ANY($1::uuid[])",
                            row["retrieved_chunk_ids"],
                        )
                        context_chunks = [c["content"] for c in chunk_rows]
                        judge = EvaluationJudge(client=client, db_pool=db_pool)
                        eval_score = await judge.evaluate_run(
                            run_id=row["run_id"],
                            query=user_msg,
                            answer=assistant_msg,
                            context=context_chunks,
                        )
                        faithfulness = eval_score.faithfulness
                    except Exception as exc:
                        logger.warning("Failed to re-evaluate faithfulness for pair %s: %s", pair_id, exc)

                is_valid, reason = self.validate_pair(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    faithfulness=faithfulness,
                )

                if is_valid:
                    valid_pairs.append({
                        "pair_id": str(pair_id),
                        "run_id": str(row["run_id"]),
                        "quality_score": row["quality_score"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                    })
                else:
                    rejected_pairs.append({
                        "pair_id": str(pair_id),
                        "run_id": str(row["run_id"]),
                        "reason": reason,
                    })

        # Write validated pairs to JSONL file
        output_file = self.training_data_dir / f"{job_uuid}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in valid_pairs:
                line = json.dumps({"messages": item["messages"]})
                f.write(line + "\n")

        return valid_pairs, rejected_pairs, output_file

    async def preview_eligible_pairs(
        self,
        db_pool,
        limit: int = 5,
        min_quality_score: float = 0.82,
    ) -> list[dict[str, Any]]:
        """Return preview of currently eligible training pairs without creating or submitting a job."""
        preview_items: list[dict[str, Any]] = []

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tp.id, tp.run_id, tp.system_prompt, tp.user_message, tp.assistant_message,
                       tp.quality_score, tp.created_at,
                       e.faithfulness, e.user_rating
                FROM training_pairs tp
                LEFT JOIN evaluations e ON e.run_id = tp.run_id
                WHERE tp.included_in_job IS NULL
                  AND tp.quality_score >= $1
                  AND (e.user_rating IS NULL OR e.user_rating >= 4)
                ORDER BY tp.created_at ASC
                LIMIT 50
                """,
                min_quality_score,
            )

            for row in rows:
                system_prompt = row["system_prompt"] or "You are a precise research assistant."
                user_msg = row["user_message"]
                assistant_msg = row["assistant_message"]
                faithfulness = row["faithfulness"]

                is_valid, _ = self.validate_pair(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    faithfulness=faithfulness,
                )

                if is_valid:
                    preview_items.append({
                        "pair_id": str(row["id"]),
                        "run_id": str(row["run_id"]),
                        "quality_score": row["quality_score"],
                        "faithfulness": faithfulness,
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                    })
                    if len(preview_items) >= limit:
                        break

        return preview_items

    async def extract_dpo_pairs(
        self,
        db_pool,
        job_id: UUID | str | None = None,
        min_chosen_rating: int = 4,
        max_rejected_rating: int = 2,
    ) -> tuple[list[dict[str, Any]], Path | None]:
        """Extract DPO preference pairs where both a chosen (rating >= 4) and rejected (rating <= 2) response exist for the same query.

        Format: {"prompt": query, "chosen": good_response, "rejected": bad_response}
        """
        dpo_pairs: list[dict[str, Any]] = []

        async with db_pool.acquire() as conn:
            # Query pairs of runs with the exact same query but differing ratings
            rows = await conn.fetch(
                """
                SELECT r1.query AS prompt,
                       r1.generation AS chosen,
                       e1.user_rating AS chosen_rating,
                       r2.generation AS rejected,
                       e2.user_rating AS rejected_rating
                FROM pipeline_runs r1
                JOIN evaluations e1 ON e1.run_id = r1.id
                JOIN pipeline_runs r2 ON LOWER(TRIM(r2.query)) = LOWER(TRIM(r1.query)) AND r1.id != r2.id
                JOIN evaluations e2 ON e2.run_id = r2.id
                WHERE e1.user_rating >= $1
                  AND e2.user_rating <= $2
                  AND r1.generation IS NOT NULL AND TRIM(r1.generation) != ''
                  AND r2.generation IS NOT NULL AND TRIM(r2.generation) != ''
                ORDER BY r1.created_at DESC
                """,
                min_chosen_rating,
                max_rejected_rating,
            )

            seen_prompts = set()
            for r in rows:
                prompt_text = (r["prompt"] or "").strip()
                chosen_text = (r["chosen"] or "").strip()
                rejected_text = (r["rejected"] or "").strip()

                if not prompt_text or not chosen_text or not rejected_text:
                    continue
                if chosen_text == rejected_text:
                    continue
                if self.check_pii(prompt_text):
                    continue

                # Deduplicate same prompt pair
                prompt_key = prompt_text.lower()
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)

                dpo_pairs.append({
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                })

        output_file = None
        if job_id is not None:
            job_uuid = UUID(str(job_id))
            output_file = self.training_data_dir / f"dpo_{job_uuid}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for pair in dpo_pairs:
                    line = json.dumps({
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                    })
                    f.write(line + "\n")

        return dpo_pairs, output_file

    async def preview_dpo_pairs(
        self,
        db_pool,
        limit: int = 5,
        min_chosen_rating: int = 4,
        max_rejected_rating: int = 2,
    ) -> list[dict[str, Any]]:
        """Return preview of eligible DPO preference pairs without writing to disk."""
        pairs, _ = await self.extract_dpo_pairs(
            db_pool=db_pool,
            job_id=None,
            min_chosen_rating=min_chosen_rating,
            max_rejected_rating=max_rejected_rating,
        )
        return pairs[:limit]

