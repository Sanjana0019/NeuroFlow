from pathlib import Path

from pipelines.ingestion.extractors.text_extractor import TextExtractor


FIXTURE_PATH = Path("backend/tests/fixtures/test_document.txt")


def test_text_extractor():
    extractor = TextExtractor()

    pages = extractor.extract(FIXTURE_PATH)

    assert len(pages) == 1

    page = pages[0]

    assert page.page_number == 1
    assert page.content_type == "text"
    assert "NeuroFlow text extraction test." in page.content
    assert "Task 4" in page.content

    assert page.metadata["source"] == "text"
    assert page.metadata["filename"] == "test_document.txt"