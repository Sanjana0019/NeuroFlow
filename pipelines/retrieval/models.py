from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

QueryType = Literal["factual", "analytical", "comparative", "procedural"]
RetrievalSource = Literal["dense", "sparse", "metadata", "fused", "reranked"]


@dataclass
class ProcessedQuery:
    """Structured representation of an analyzed user query."""

    original_query: str
    expanded_queries: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    query_type: QueryType = "factual"


@dataclass
class RetrievalResult:
    """Standardized retrieval candidate chunk."""

    chunk_id: UUID | str
    document_id: UUID | str
    content: str
    score: float = 0.0
    rank: int = 1
    source: RetrievalSource = "dense"
    filename: str = ""
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Formatted context ready for LLM consumption with token budget accounting."""

    context: str
    chunks_used: list[RetrievalResult] = field(default_factory=list)
    total_tokens: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
