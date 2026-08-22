from pathlib import Path

from pipelines.ingestion.extractors.csv_extractor import CSVExtractor


FIXTURE_PATH = Path("backend/tests/fixtures/test_data.csv")


def test_csv_extractor():
    extractor = CSVExtractor()

    pages = extractor.extract(FIXTURE_PATH)

    assert len(pages) == 1

    page = pages[0]

    assert page.page_number == 1
    assert page.content_type == "table"

    assert "Component" in page.content
    assert "Status" in page.content
    assert "PDF Extractor" in page.content
    assert "Working" in page.content

    assert page.metadata["source"] == "csv"
    assert page.metadata["rows"] == 3
    assert page.metadata["columns"] == 2
    assert page.metadata["column_names"] == ["Component", "Status"]