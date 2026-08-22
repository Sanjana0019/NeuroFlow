from dataclasses import dataclass, field
from typing import Any, Literal


ContentType = Literal["text", "table", "image_description"]


@dataclass
class ExtractedPage:
    """Normalized representation of extracted document content."""

    page_number: int
    content: str
    content_type: ContentType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Normalized chunk of document content ready for embedding and persistence."""

    content: str
    chunk_index: int
    token_count: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
