import pytest

from pipelines.ingestion.extractors.url_extractor import URLExtractor
from pipelines.ingestion.models import ExtractedPage


class FakeHTTPResponse:
    def __init__(self, text: str, status_code: int = 200, url: str = "https://example.com/article"):
        self.text = text
        self.status_code = status_code
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("Error", request=None, response=self)


def test_url_extractor_real_or_fixture():
    """Extract page content and metadata from example.com."""
    extractor = URLExtractor()
    pages = extractor.extract("https://example.com")

    assert len(pages) == 1
    page = pages[0]

    assert isinstance(page, ExtractedPage)
    assert page.page_number == 1
    assert page.content_type == "text"
    assert len(page.content) > 0
    assert "documentation examples" in page.content
    assert page.metadata["source"] == "url"
    assert page.metadata["status_code"] == 200
    assert page.metadata["url"] == "https://example.com"


def test_robots_txt_disallow_blocks_fetch(monkeypatch):
    """When robots.txt disallows the path, URLExtractor raises ValueError."""
    def fake_get(url, **kwargs):
        if url.endswith("/robots.txt"):
            return FakeHTTPResponse(
                text="User-agent: *\nDisallow: /private/\n",
                status_code=200,
            )
        return FakeHTTPResponse(text="<html><body>Secret content</body></html>")

    monkeypatch.setattr("httpx.get", fake_get)

    extractor = URLExtractor()
    with pytest.raises(ValueError, match="disallowed by robots.txt"):
        extractor.extract("https://testsite.org/private/doc.html")


def test_robots_txt_allow_proceeds_fetch(monkeypatch):
    """When robots.txt permits the path, URLExtractor successfully extracts content and metadata."""
    def fake_get(url, **kwargs):
        if url.endswith("/robots.txt"):
            return FakeHTTPResponse(
                text="User-agent: *\nDisallow: /private/\nAllow: /public/\n",
                status_code=200,
            )
        return FakeHTTPResponse(
            text="<html><head><title>Public Article</title><meta name='author' content='Jane Doe'/></head><body><article><p>This is readable public content.</p></article></body></html>",
            url="https://testsite.org/public/doc.html",
        )

    monkeypatch.setattr("httpx.get", fake_get)

    extractor = URLExtractor()
    pages = extractor.extract("https://testsite.org/public/doc.html")

    assert len(pages) == 1
    assert "readable public content" in pages[0].content
    assert pages[0].metadata["source"] == "url"
    assert pages[0].metadata["title"] == "Public Article"
    assert pages[0].metadata["author"] == "Jane Doe"


def test_robots_txt_error_handles_safely(monkeypatch):
    """When robots.txt is missing (404) or throws error, extraction proceeds gracefully."""
    def fake_get(url, **kwargs):
        if url.endswith("/robots.txt"):
            return FakeHTTPResponse(text="Not found", status_code=404)
        return FakeHTTPResponse(
            text="<html><body><article><p>Default open content when robots.txt is 404.</p></article></body></html>",
            url="https://open.org/page",
        )

    monkeypatch.setattr("httpx.get", fake_get)

    extractor = URLExtractor()
    pages = extractor.extract("https://open.org/page")

    assert len(pages) == 1
    assert "Default open content" in pages[0].content