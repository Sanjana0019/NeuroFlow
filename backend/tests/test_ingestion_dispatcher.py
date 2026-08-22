from pathlib import Path

from pipelines.ingestion.dispatcher import IngestionDispatcher


FIXTURE_DIR = Path("backend/tests/fixtures")


def test_dispatch_csv():
    dispatcher = IngestionDispatcher()

    pages = dispatcher.dispatch(
        str(FIXTURE_DIR / "test_data.csv")
    )

    assert len(pages) == 1
    assert pages[0].content_type == "table"
    assert pages[0].metadata["source"] == "csv"


def test_dispatch_text():
    dispatcher = IngestionDispatcher()

    pages = dispatcher.dispatch(
        str(FIXTURE_DIR / "test_document.txt")
    )

    assert len(pages) == 1
    assert pages[0].content_type == "text"
    assert pages[0].metadata["source"] == "text"


def test_dispatch_pptx():
    dispatcher = IngestionDispatcher()

    pages = dispatcher.dispatch(
        str(FIXTURE_DIR / "test_presentation.pptx")
    )

    assert len(pages) == 2
    assert pages[0].metadata["source"] == "pptx"


def test_dispatch_url():
    dispatcher = IngestionDispatcher()

    pages = dispatcher.dispatch("https://example.com")

    assert len(pages) == 1
    assert pages[0].content_type == "text"
    assert pages[0].metadata["source"] == "url"


def test_dispatch_unsupported_extension():
    dispatcher = IngestionDispatcher()

    unsupported_file = FIXTURE_DIR / "unsupported.xyz"
    unsupported_file.touch()

    try:
        dispatcher.dispatch(str(unsupported_file))
        assert False, "Expected ValueError for unsupported extension"
    except ValueError as exc:
        assert "Unsupported ingestion source type" in str(exc)