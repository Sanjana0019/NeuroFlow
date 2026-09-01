import json
import logging
import re
from typing import Any

from backend.providers.base import ChatMessage
from backend.providers.router import RoutingCriteria
from pipelines.retrieval.models import ProcessedQuery, QueryType

logger = logging.getLogger("neuroflow.retrieval.query_processor")

SYSTEM_PROMPT = """You are an advanced query understanding assistant for an enterprise RAG retrieval pipeline.
Analyze the user's input query and return a valid JSON object containing:
1. "expanded_queries": A list of 2 to 3 alternative search queries or paraphrasings that capture different keywords/perspectives for dense/sparse retrieval.
2. "metadata_filters": A JSON object of explicit or implicit metadata filters detected in the query (e.g. {"year": 2023, "author": "...", "category": "...", "topic": "..."}). If none, return {}.
3. "query_type": Exactly one of ["factual", "analytical", "comparative", "procedural"].

Respond with strictly valid JSON only. Do not include markdown codeblocks or extra text.

Example input: "Show me 2023 climate change reports"
Example output:
{
  "expanded_queries": ["global warming environmental assessment 2023", "climate policy impact analysis 2023"],
  "metadata_filters": {"year": 2023, "topic": "climate"},
  "query_type": "analytical"
}
"""


class QueryProcessor:
    """Processes user queries with expansion, metadata filter extraction, and query classification."""

    def __init__(self, client=None):
        self.client = client

    async def process(self, query: str) -> ProcessedQuery:
        """Analyze query to extract expansions, metadata filters, and classification."""
        clean_query = (query or "").strip()
        if not clean_query:
            return ProcessedQuery(
                original_query="",
                expanded_queries=[],
                metadata_filters={},
                query_type="factual",
            )

        if not self.client:
            # Fallback heuristic processing when client is not supplied
            return self._fallback_processing(clean_query)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=clean_query),
        ]

        try:
            criteria = RoutingCriteria(task_type="query_processing")
            result = await asyncio.wait_for(
                self.client.chat(messages=messages, routing_criteria=criteria),
                timeout=2.0,
            )
            raw_text = result.content.strip()

            # Clean markdown codeblocks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)

            expansions = parsed.get("expanded_queries", [])
            if not isinstance(expansions, list):
                expansions = []
            expansions = [str(q).strip() for q in expansions if str(q).strip() and str(q).strip() != clean_query]

            metadata_filters = parsed.get("metadata_filters", {})
            if not isinstance(metadata_filters, dict):
                metadata_filters = {}

            q_type = str(parsed.get("query_type", "factual")).lower()
            valid_types = {"factual", "analytical", "comparative", "procedural"}
            if q_type not in valid_types:
                q_type = "factual"

            return ProcessedQuery(
                original_query=clean_query,
                expanded_queries=expansions,
                metadata_filters=metadata_filters,
                query_type=q_type,  # type: ignore
            )

        except Exception as exc:
            logger.warning("Query processor LLM extraction failed: %s. Using fallback.", exc)
            return self._fallback_processing(clean_query)

    def _fallback_processing(self, query: str) -> ProcessedQuery:
        """Heuristic fallback for query classification and filter detection."""
        lower = query.lower()

        # Query classification heuristic
        if any(w in lower for vs, w in [("c", "compare"), ("c", "versus"), ("c", "vs"), ("c", "difference")]):
            q_type: QueryType = "comparative"
        elif any(w in lower for vs, w in [("p", "how to"), ("p", "steps"), ("p", "procedure"), ("p", "guide")]):
            q_type = "procedural"
        elif any(w in lower for vs, w in [("a", "why"), ("a", "analyze"), ("a", "impact"), ("a", "trend")]):
            q_type = "analytical"
        else:
            q_type = "factual"

        # Year filter heuristic
        filters: dict[str, Any] = {}
        year_match = re.search(r"\b(20\d\d|19\d\d)\b", query)
        if year_match:
            try:
                filters["year"] = int(year_match.group(1))
            except ValueError:
                pass

        if "climate" in lower:
            filters["topic"] = "climate"

        return ProcessedQuery(
            original_query=query,
            expanded_queries=[],
            metadata_filters=filters,
            query_type=q_type,
        )
