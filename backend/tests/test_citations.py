from uuid import uuid4
import pytest

from pipelines.generation.citations import CitationParser
from pipelines.retrieval.models import RetrievalResult


def test_valid_citation_mapping():
    """Maps [Source 1] to the exact first chunk in chunks_used."""
    c1_id = uuid4()
    chunks = [
        RetrievalResult(
            chunk_id=c1_id,
            document_id=uuid4(),
            content="Self-attention mechanisms allow sequence modeling without recurrence.",
            filename="attention.pdf",
            page_number=3,
        )
    ]

    response = "Based on [Source 1], attention models sequence dependencies."
    citations = CitationParser.parse(response, chunks)

    assert len(citations) == 1
    cit = citations[0]
    assert cit.reference == "[Source 1]"
    assert cit.chunk_id == c1_id
    assert cit.document_name == "attention.pdf"
    assert cit.page_number == 3
    assert "Self-attention mechanisms" in cit.content_preview
    assert cit.invalid_citation is False


def test_multiple_and_repeated_citations():
    """Extracts multiple distinct citations in order and deduplicates repeats."""
    c1_id = uuid4()
    c2_id = uuid4()
    chunks = [
        RetrievalResult(chunk_id=c1_id, document_id=uuid4(), content="First chunk content", filename="doc1.pdf"),
        RetrievalResult(chunk_id=c2_id, document_id=uuid4(), content="Second chunk content", filename="doc2.pdf"),
    ]

    response = "Fact A is from [Source 1]. Fact B is from [Source 2]. Another note on [Source 1]."
    citations = CitationParser.parse(response, chunks)

    assert len(citations) == 2
    assert citations[0].reference == "[Source 1]"
    assert citations[0].chunk_id == c1_id
    assert citations[1].reference == "[Source 2]"
    assert citations[1].chunk_id == c2_id


def test_invalid_hallucinated_citation_detected():
    """Flags out-of-bounds citations as invalid_citation=True."""
    chunks = [
        RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content="Only chunk", filename="doc1.pdf"),
    ]

    # Only 1 chunk exists in context, but response hallucinated [Source 99]
    response = "According to [Source 99], something ungrounded happened."
    citations = CitationParser.parse(response, chunks)

    assert len(citations) == 1
    cit = citations[0]
    assert cit.reference == "[Source 99]"
    assert cit.invalid_citation is True
    assert cit.chunk_id is None


def test_content_preview_character_limit():
    """Preview is capped at the first 100 characters of chunk content."""
    long_content = "A" * 250
    chunks = [
        RetrievalResult(chunk_id=uuid4(), document_id=uuid4(), content=long_content, filename="long.pdf"),
    ]

    citations = CitationParser.parse("Reference [Source 1].", chunks)
    assert len(citations) == 1
    assert len(citations[0].content_preview) <= 100
    assert citations[0].content_preview == "A" * 100
