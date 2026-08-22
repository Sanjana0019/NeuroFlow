import json
import pytest

from backend.providers.base import GenerationResult
from pipelines.retrieval.query_processor import QueryProcessor


class FakeLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def chat(self, messages, routing_criteria, **kwargs):
        self.calls.append((messages, routing_criteria))
        return GenerationResult(
            content=self.response_text,
            model="gpt-4o-mini",
            input_tokens=50,
            output_tokens=30,
            latency_ms=120.0,
            cost_usd=0.0001,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_query_processor_with_mock_llm():
    """QueryProcessor parses JSON output for expansion, filters, and query classification."""
    mock_payload = {
        "expanded_queries": [
            "transformer self-attention mechanism explanation",
            "calculating attention weights in multi-head attention",
        ],
        "metadata_filters": {"year": 2023, "topic": "transformers"},
        "query_type": "analytical",
    }
    client = FakeLLMClient(json.dumps(mock_payload))
    processor = QueryProcessor(client=client)

    result = await processor.process("How does attention work in transformers?")

    assert result.original_query == "How does attention work in transformers?"
    assert len(result.expanded_queries) == 2
    assert "transformer self-attention mechanism explanation" in result.expanded_queries
    assert result.metadata_filters == {"year": 2023, "topic": "transformers"}
    assert result.query_type == "analytical"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_query_processor_markdown_json_fences():
    """QueryProcessor cleans markdown ```json fences if returned by LLM."""
    mock_json = """```json
    {
      "expanded_queries": ["climate policy 2023"],
      "metadata_filters": {"year": 2023, "topic": "climate"},
      "query_type": "factual"
    }
    ```"""
    client = FakeLLMClient(mock_json)
    processor = QueryProcessor(client=client)

    result = await processor.process("Show me 2023 climate documents")
    assert result.metadata_filters["year"] == 2023
    assert result.metadata_filters["topic"] == "climate"
    assert result.query_type == "factual"


@pytest.mark.asyncio
async def test_query_processor_fallback_without_client():
    """Fallback heuristics handle year, topic, and classification when no LLM client is configured."""
    processor = QueryProcessor(client=None)

    # Analytical query with year
    res1 = await processor.process("Why did inflation rise in 2023?")
    assert res1.query_type == "analytical"
    assert res1.metadata_filters.get("year") == 2023

    # Comparative query
    res2 = await processor.process("Compare BERT versus GPT models")
    assert res2.query_type == "comparative"

    # Procedural query
    res3 = await processor.process("How to install pgvector on PostgreSQL?")
    assert res3.query_type == "procedural"

    # Empty query
    res4 = await processor.process("   ")
    assert res4.original_query == ""
    assert res4.expanded_queries == []
