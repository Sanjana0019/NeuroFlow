from uuid import uuid4

import pytest
import tiktoken

from pipelines.ingestion.chunker import Chunker
from pipelines.ingestion.models import ExtractedPage


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


# =====================================================================
# 1. Fixed-size Strategy & Basic Tests
# =====================================================================


def test_short_page_produces_single_chunk():
    """A short page within chunk_size should produce exactly one chunk."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    page = ExtractedPage(
        page_number=1,
        content="This is a short document page with just a single sentence.",
        content_type="text",
        metadata={"source": "test_doc.pdf"},
    )

    chunks = chunker.chunk([page])

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].content == page.content
    assert chunks[0].metadata["source"] == "test_doc.pdf"
    assert chunks[0].metadata["content_type"] == "text"
    assert chunks[0].metadata["strategy"] == "fixed_size"


def test_token_count_matches_tiktoken(tokenizer):
    """Chunk token_count should exactly match tiktoken encoding count."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    sample_text = (
        "NeuroFlow is an enterprise RAG system that uses PostgreSQL and pgvector."
    )
    page = ExtractedPage(
        page_number=1,
        content=sample_text,
        content_type="text",
    )

    chunks = chunker.chunk([page])

    expected_tokens = len(tokenizer.encode(sample_text))
    assert len(chunks) == 1
    assert chunks[0].token_count == expected_tokens
    assert chunker.count_tokens(sample_text) == expected_tokens


def test_metadata_propagation_and_optional_document_id():
    """Chunk metadata should inherit page metadata and document_id when supplied."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    doc_id = str(uuid4())
    page = ExtractedPage(
        page_number=3,
        content="Content from page 3 with custom headers.",
        content_type="text",
        metadata={"author": "Sanjana", "category": "engineering"},
    )

    # With document_id
    chunks_with_doc = chunker.chunk([page], document_id=doc_id)
    assert len(chunks_with_doc) == 1
    assert chunks_with_doc[0].metadata["document_id"] == doc_id
    assert chunks_with_doc[0].metadata["author"] == "Sanjana"
    assert chunks_with_doc[0].metadata["category"] == "engineering"
    assert chunks_with_doc[0].metadata["page_number"] == 3

    # Without document_id (should NOT invent one)
    chunks_without_doc = chunker.chunk([page])
    assert len(chunks_without_doc) == 1
    assert "document_id" not in chunks_without_doc[0].metadata
    assert chunks_without_doc[0].metadata["author"] == "Sanjana"


def test_multiple_pages_global_sequential_chunk_index():
    """chunk_index must be globally sequential across pages and not reset per page."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    pages = [
        ExtractedPage(page_number=1, content="First page content.", content_type="text"),
        ExtractedPage(page_number=2, content="Second page content.", content_type="text"),
        ExtractedPage(page_number=3, content="Third page content.", content_type="text"),
    ]

    chunks = chunker.chunk(pages)

    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.page_number for c in chunks] == [1, 2, 3]


def test_long_content_split_and_token_limit_respected():
    """Content exceeding chunk_size produces multiple chunks, none exceeding limit."""
    chunk_size = 50
    overlap = 10
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap)

    paragraphs = [
        f"Paragraph {i}: This is section {i} containing informative text for testing chunking boundaries."
        for i in range(20)
    ]
    long_content = "\n\n".join(paragraphs)

    page = ExtractedPage(page_number=1, content=long_content, content_type="text")
    chunks = chunker.chunk([page])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunk_size
        assert chunker.count_tokens(chunk.content) <= chunk_size


def test_overlap_applied_in_sequential_chunks():
    """Token-based splitting should maintain configured overlap between consecutive windows."""
    chunk_size = 30
    overlap = 10
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap)

    words = [f"token{i}" for i in range(100)]
    continuous_text = " ".join(words)
    page = ExtractedPage(page_number=1, content=continuous_text, content_type="text")

    chunks = chunker.chunk([page])

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        assert chunker.count_tokens(chunks[i].content) <= chunk_size
        assert chunker.count_tokens(chunks[i + 1].content) <= chunk_size

        words_current = chunks[i].content.split()
        words_next = chunks[i + 1].content.split()

        shared_overlap = set(words_current[-overlap:]).intersection(set(words_next[:overlap]))
        assert len(shared_overlap) > 0


def test_empty_and_whitespace_pages_produce_no_chunks():
    """Empty or whitespace-only pages should be ignored and produce zero chunks."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    pages = [
        ExtractedPage(page_number=1, content="", content_type="text"),
        ExtractedPage(page_number=2, content="   \n\t   ", content_type="text"),
        ExtractedPage(page_number=3, content="Valid content on page 3", content_type="text"),
    ]

    chunks = chunker.chunk(pages)

    assert len(chunks) == 1
    assert chunks[0].content == "Valid content on page 3"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize(
    "size,overlap",
    [
        (0, 10),
        (-50, 10),
        (512, -1),
        (512, -10),
        (512, 512),
        (512, 600),
    ],
)
def test_invalid_configuration_raises_value_error(size, overlap):
    """Invalid chunk_size or chunk_overlap values must raise ValueError."""
    with pytest.raises(ValueError):
        Chunker(chunk_size=size, chunk_overlap=overlap)


def test_table_content_preserved():
    """Table content should retain row structure and format."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    table_content = (
        "| ID | Name    | Department  |\n"
        "|----|---------|-------------|\n"
        "| 1  | Alice   | Engineering |\n"
        "| 2  | Bob     | Research    |\n"
        "| 3  | Charlie | Product     |"
    )

    page = ExtractedPage(
        page_number=1,
        content=table_content,
        content_type="table",
        metadata={"format": "markdown_table"},
    )

    chunks = chunker.chunk([page])

    assert len(chunks) == 1
    assert chunks[0].content == table_content
    assert chunks[0].metadata["content_type"] == "table"
    assert chunks[0].metadata["format"] == "markdown_table"
    assert "| Alice   | Engineering |" in chunks[0].content


# =====================================================================
# 2. Automatic Strategy Selection Tests
# =====================================================================


def test_auto_selection_table_uses_fixed_size():
    """Pages with content_type == 'table' select fixed_size strategy."""
    chunker = Chunker()
    page = ExtractedPage(page_number=1, content="| Col1 | Col2 |\n|---|---|\n| A | B |", content_type="table")
    strategy = chunker.select_strategy([page])
    assert strategy == "fixed_size"


def test_auto_selection_docx_with_headings_uses_hierarchical():
    """DOCX with heading levels/sections selects hierarchical strategy."""
    chunker = Chunker()
    pages = [
        ExtractedPage(page_number=1, content="Introduction", content_type="text", metadata={"source": "paragraph", "level": "h1", "section": "Introduction"}),
        ExtractedPage(page_number=2, content="Some introductory text.", content_type="text", metadata={"source": "paragraph"}),
    ]
    strategy = chunker.select_strategy(pages)
    assert strategy == "hierarchical"


def test_auto_selection_pdf_large_page_count_uses_semantic():
    """PDF documents with > 50 pages select semantic strategy."""
    chunker = Chunker()
    pages = [
        ExtractedPage(page_number=i, content=f"Content for page {i}", content_type="text", metadata={"source": "pdf", "page": i})
        for i in range(1, 55)
    ]
    strategy = chunker.select_strategy(pages)
    assert strategy == "semantic"


def test_auto_selection_default_uses_fixed_size():
    """Standard text pages default to fixed_size strategy."""
    chunker = Chunker()
    pages = [
        ExtractedPage(page_number=1, content="Short plain document.", content_type="text", metadata={"source": "text"}),
    ]
    strategy = chunker.select_strategy(pages)
    assert strategy == "fixed_size"


# =====================================================================
# 3. Explicit Hierarchical Strategy Tests
# =====================================================================


def test_hierarchical_strategy_parent_child_metadata():
    """Hierarchical chunking tracks parent section, heading level, and path."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    pages = [
        ExtractedPage(
            page_number=1,
            content="1. System Architecture",
            content_type="text",
            metadata={"source": "paragraph", "level": "h1", "section": "1. System Architecture"},
        ),
        ExtractedPage(
            page_number=2,
            content="NeuroFlow uses a layered pipeline design for ingestion and retrieval.",
            content_type="text",
            metadata={"source": "paragraph"},
        ),
        ExtractedPage(
            page_number=3,
            content="1.1 Ingestion Pipeline",
            content_type="text",
            metadata={"source": "paragraph", "level": "h2", "section": "1.1 Ingestion Pipeline"},
        ),
        ExtractedPage(
            page_number=4,
            content="The ingestion pipeline processes multimodal files into vectors.",
            content_type="text",
            metadata={"source": "paragraph"},
        ),
    ]

    chunks = chunker.chunk(pages, strategy="hierarchical")

    assert len(chunks) == 4
    for c in chunks:
        assert c.metadata["strategy"] == "hierarchical"

    # Chunk 1 (Intro content under 1. System Architecture)
    assert chunks[1].metadata["parent_section"] == "1. System Architecture"
    assert chunks[1].metadata["heading_level"] == "h1"
    assert chunks[1].metadata["hierarchy_path"] == ["1. System Architecture"]

    # Chunk 3 (Content under 1.1 Ingestion Pipeline)
    assert chunks[3].metadata["parent_section"] == "1.1 Ingestion Pipeline"
    assert chunks[3].metadata["heading_level"] == "h2"
    assert chunks[3].metadata["hierarchy_path"] == ["1. System Architecture", "1.1 Ingestion Pipeline"]


# =====================================================================
# 4. Explicit Semantic Strategy Tests
# =====================================================================


def test_semantic_strategy_splits_on_low_similarity():
    """Semantic strategy splits chunks when adjacent sentence similarity drops below threshold (0.7)."""
    chunker = Chunker(chunk_size=512, chunk_overlap=64)

    # 4 sentences: (S1, S2 are topic A; S3, S4 are topic B)
    text = (
        "PostgreSQL is an advanced relational database. "
        "It supports relational tables and pgvector indices. "
        "Quantum computing relies on quantum bits called qubits. "
        "Superposition and entanglement are fundamental quantum properties."
    )
    page = ExtractedPage(page_number=1, content=text, content_type="text")

    # Mock embeddings: topic A ~ [1.0, 0.0], topic B ~ [0.0, 1.0]
    # S1 vs S2 similarity = 1.0 (>= 0.7 -> stay together)
    # S2 vs S3 similarity = 0.0 (< 0.7 -> split!)
    # S3 vs S4 similarity = 1.0 (>= 0.7 -> stay together)
    sentence_embeddings = [
        [1.0, 0.0],  # S1 (PostgreSQL)
        [1.0, 0.0],  # S2 (PostgreSQL)
        [0.0, 1.0],  # S3 (Quantum)
        [0.0, 1.0],  # S4 (Quantum)
    ]

    chunks = chunker.chunk_semantic(
        [page],
        similarity_threshold=0.7,
        sentence_embeddings=sentence_embeddings,
    )

    assert len(chunks) == 2
    assert "PostgreSQL" in chunks[0].content
    assert "pgvector" in chunks[0].content
    assert "Quantum computing" in chunks[1].content
    assert "Superposition" in chunks[1].content
    assert chunks[0].metadata["strategy"] == "semantic"
    assert chunks[1].metadata["strategy"] == "semantic"


def test_semantic_strategy_fails_when_embedder_missing():
    """Semantic strategy raises clear RuntimeError if no embedding client is available."""
    chunker = Chunker()
    page = ExtractedPage(
        page_number=1,
        content="Sentence 1. Sentence 2. Sentence 3.",
        content_type="text",
    )

    with pytest.raises(RuntimeError, match="Semantic chunking requires an available embedding provider"):
        chunker.chunk_semantic([page])
