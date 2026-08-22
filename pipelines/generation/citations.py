from dataclasses import dataclass
import re
from typing import Any
from uuid import UUID

from pipelines.retrieval.models import RetrievalResult


@dataclass
class Citation:
    """Represents a validated citation mapped to an actual context chunk."""

    reference: str
    chunk_id: UUID | str | None
    document_name: str
    page_number: int | None
    content_preview: str
    invalid_citation: bool = False


class CitationParser:
    """Parses and validates [Source N] references against supplied context chunks."""

    SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

    @classmethod
    def parse(
        cls,
        response_text: str,
        chunks_used: list[RetrievalResult] | list[dict[str, Any]],
    ) -> list[Citation]:
        """Extract [Source N] references and map to actual chunks used in context."""
        if not response_text:
            return []

        # Find all [Source N] matches
        matches = cls.SOURCE_PATTERN.finditer(response_text)
        seen_indices: set[int] = set()
        citations: list[Citation] = []

        total_chunks = len(chunks_used)

        for match in matches:
            source_num = int(match.group(1))
            reference_str = f"[Source {source_num}]"

            if source_num in seen_indices:
                continue
            seen_indices.add(source_num)

            # Validate whether source number exists in context chunks (1-indexed)
            if 1 <= source_num <= total_chunks:
                chunk_item = chunks_used[source_num - 1]

                if isinstance(chunk_item, RetrievalResult):
                    c_id = chunk_item.chunk_id
                    doc_name = chunk_item.filename or "document"
                    page_num = chunk_item.page_number
                    content_str = chunk_item.content
                elif isinstance(chunk_item, dict):
                    c_id = chunk_item.get("chunk_id") or chunk_item.get("id")
                    doc_name = chunk_item.get("filename") or chunk_item.get("document_name") or "document"
                    page_num = chunk_item.get("page_number")
                    content_str = chunk_item.get("content", "")
                else:
                    c_id = getattr(chunk_item, "chunk_id", None) or getattr(chunk_item, "id", None)
                    doc_name = getattr(chunk_item, "filename", "document")
                    page_num = getattr(chunk_item, "page_number", None)
                    content_str = getattr(chunk_item, "content", "")

                preview = content_str[:100].strip() if content_str else ""

                citations.append(
                    Citation(
                        reference=reference_str,
                        chunk_id=c_id,
                        document_name=doc_name,
                        page_number=page_num,
                        content_preview=preview,
                        invalid_citation=False,
                    )
                )
            else:
                # Flag hallucinated / out-of-bounds citations
                citations.append(
                    Citation(
                        reference=reference_str,
                        chunk_id=None,
                        document_name="unknown",
                        page_number=None,
                        content_preview="",
                        invalid_citation=True,
                    )
                )

        return citations
