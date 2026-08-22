import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from opentelemetry import trace

from backend.providers.router import RoutingCriteria
from evaluation.metrics.answer_relevance import evaluate_answer_relevance
from evaluation.metrics.context_precision import evaluate_context_precision
from evaluation.metrics.context_recall import evaluate_context_recall
from evaluation.metrics.faithfulness import evaluate_faithfulness

logger = logging.getLogger("neuroflow.evaluation.judge")
tracer = trace.get_tracer("neuroflow.evaluation")


@dataclass
class EvaluationScore:
    """Consolidated evaluation metrics for a single RAG pipeline run."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    judge_model: str
    is_training_candidate: bool = False


class EvaluationJudge:
    """Orchestrates automated evaluation of generated answers using RAGAS metrics and OpenTelemetry."""

    def __init__(self, client: Any = None, db_pool: Any = None):
        self.client = client
        self.db_pool = db_pool

    async def evaluate_run(
        self,
        run_id: UUID | str,
        query: str,
        answer: str,
        context: str | list[str],
        system_prompt: str | None = None,
        judge_model_override: str | None = None,
    ) -> EvaluationScore:
        """Run all 4 metrics concurrently via asyncio.gather, persist to DB, and trace with OpenTelemetry."""
        run_uuid = UUID(str(run_id))

        if isinstance(context, list):
            chunks_list = [str(c) for c in context]
            context_str = "\n\n".join(chunks_list)
        else:
            context_str = str(context)
            chunks_list = [c.strip() for c in context_str.split("\n\n") if c.strip()]

        judge_model = judge_model_override or "gpt-4o-mini"
        if self.client and hasattr(self.client, "router"):
            try:
                criteria = RoutingCriteria(task_type="evaluation")
                routed = await self.client.router.route(criteria)
                judge_model = routed.get("model", judge_model)
            except Exception:
                pass

        with tracer.start_as_current_span("evaluation.judge") as span:
            span.set_attribute("run_id", str(run_uuid))
            span.set_attribute("judge_model", judge_model)

            # 1. Execute all 4 metrics concurrently
            faithfulness_task = evaluate_faithfulness(
                query=query,
                answer=answer,
                context=context_str,
                client=self.client,
            )
            relevance_task = evaluate_answer_relevance(
                query=query,
                answer=answer,
                client=self.client,
            )
            precision_task = evaluate_context_precision(
                query=query,
                chunks=chunks_list,
                answer=answer,
                client=self.client,
            )
            recall_task = evaluate_context_recall(
                query=query,
                chunks=chunks_list,
                answer=answer,
                client=self.client,
            )

            f_score, r_score, p_score, rc_score = await asyncio.gather(
                faithfulness_task,
                relevance_task,
                precision_task,
                recall_task,
            )

            # 2. Composite Overall Score Calculation
            overall = (
                0.35 * f_score
                + 0.30 * r_score
                + 0.20 * p_score
                + 0.15 * rc_score
            )
            overall_score = round(max(0.0, min(1.0, overall)), 4)

            # Record OpenTelemetry Span Attributes
            span.set_attribute("faithfulness", f_score)
            span.set_attribute("answer_relevance", r_score)
            span.set_attribute("context_precision", p_score)
            span.set_attribute("context_recall", rc_score)
            span.set_attribute("overall_score", overall_score)

            is_training_candidate = overall_score > 0.8

            eval_score = EvaluationScore(
                faithfulness=f_score,
                answer_relevance=r_score,
                context_precision=p_score,
                context_recall=rc_score,
                overall_score=overall_score,
                judge_model=judge_model,
                is_training_candidate=is_training_candidate,
            )

            # 3. Persist to PostgreSQL database
            if self.db_pool is not None:
                await self._persist_evaluation(
                    run_uuid=run_uuid,
                    score=eval_score,
                    query=query,
                    answer=answer,
                    system_prompt=system_prompt,
                )

            return eval_score

    async def _persist_evaluation(
        self,
        run_uuid: UUID,
        score: EvaluationScore,
        query: str,
        answer: str,
        system_prompt: str | None = None,
    ) -> None:
        """Persist evaluation scores and insert training pairs if overall_score > 0.8."""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Insert evaluations record
                    await conn.execute(
                        """
                        INSERT INTO evaluations (
                            run_id,
                            faithfulness,
                            answer_relevance,
                            context_precision,
                            context_recall,
                            overall_score,
                            judge_model
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        run_uuid,
                        score.faithfulness,
                        score.answer_relevance,
                        score.context_precision,
                        score.context_recall,
                        score.overall_score,
                        score.judge_model,
                    )

                    # If score > 0.8, insert as training pair candidate
                    if score.is_training_candidate:
                        await conn.execute(
                            """
                            INSERT INTO training_pairs (
                                run_id,
                                system_prompt,
                                user_message,
                                assistant_message,
                                quality_score
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            run_uuid,
                            system_prompt or "You are a precise research assistant.",
                            query,
                            answer,
                            score.overall_score,
                        )
        except Exception as exc:
            logger.error("Failed to persist evaluation for run %s: %s", run_uuid, exc)
