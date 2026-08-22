from uuid import uuid4

from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.models import RetrievalResult


def test_context_assembler_formatting():
    """Assembles formatted context with source headers, page numbers, and filenames."""
    assembler = ContextAssembler(token_budget=4000)

    c1 = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Transformers utilize multi-head self-attention.",
        filename="attention_paper.pdf",
        page_number=3,
    )
    c2 = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Positional encoding injects sequence order.",
        filename="attention_paper.pdf",
        page_number=4,
    )

    assembled = assembler.assemble([c1, c2])

    assert "[Source 1 — attention_paper.pdf, page 3]" in assembled.context
    assert "Transformers utilize multi-head self-attention." in assembled.context
    assert "[Source 2 — attention_paper.pdf, page 4]" in assembled.context
    assert "Positional encoding injects sequence order." in assembled.context
    assert len(assembled.chunks_used) == 2
    assert len(assembled.sources) == 2
    assert assembled.sources[0]["source_index"] == 1
    assert assembled.sources[0]["filename"] == "attention_paper.pdf"
    assert assembled.total_tokens > 0


def test_token_budget_enforcement_never_cuts_chunks():
    """ContextAssembler stops before adding a chunk that would exceed token budget and does not split mid-chunk."""
    assembler = ContextAssembler(token_budget=30)

    # Chunk 1 (~10 tokens)
    c1 = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Small first chunk fitting in budget.",
        filename="doc1.pdf",
    )
    # Chunk 2 (~100 tokens, exceeds remaining 20 token allowance)
    c2 = RetrievalResult(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="This is a very long second chunk with lots of repeated text words that would definitely exceed thirty tokens by a wide margin if included in full.",
        filename="doc2.pdf",
    )

    assembled = assembler.assemble([c1, c2])

    assert len(assembled.chunks_used) == 1
    assert assembled.chunks_used[0].chunk_id == c1.chunk_id
    assert assembled.total_tokens <= 30
    assert "doc2.pdf" not in assembled.context


def test_empty_chunks_returns_empty_context():
    """Empty chunk list returns empty context gracefully."""
    assembler = ContextAssembler(token_budget=4000)
    assembled = assembler.assemble([])

    assert assembled.context == ""
    assert assembled.chunks_used == []
    assert assembled.total_tokens == 0
    assert assembled.sources == []
