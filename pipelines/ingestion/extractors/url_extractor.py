from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from pipelines.ingestion.models import ExtractedPage

USER_AGENT = "NeuroFlow/1.0"


class URLExtractor:
    """Extract readable text and metadata from webpages with robots.txt compliance."""

    def __init__(self, user_agent: str = USER_AGENT, check_robots: bool = True) -> None:
        self.user_agent = user_agent
        self.check_robots = check_robots

    def _is_allowed_by_robots(self, url: str, parsed_url) -> bool:
        """Check if target URL is permitted according to the host's robots.txt."""
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser()

        try:
            response = httpx.get(
                robots_url,
                timeout=5.0,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            if response.status_code == 200 and response.text:
                rp.parse(response.text.splitlines())
                return rp.can_fetch(self.user_agent, url)
            elif response.status_code in {401, 403}:
                # Explicitly restricted robots.txt is treated as disallow
                return False
        except Exception:
            # If robots.txt is unreachable (e.g. 404, network failure), default to allow
            return True

        return True

    def extract(self, url: str) -> list[ExtractedPage]:
        """Fetch and extract clean text and metadata from a web page."""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        target_url = url.strip()

        # 1. Robots.txt Compliance Check
        if self.check_robots:
            allowed = self._is_allowed_by_robots(target_url, parsed)
            if not allowed:
                raise ValueError(
                    f"Access to URL '{target_url}' is disallowed by robots.txt for User-Agent '{self.user_agent}'"
                )

        # 2. Fetch page content
        try:
            response = httpx.get(
                target_url,
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": self.user_agent,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"Failed to fetch URL '{target_url}': {exc}") from exc

        html_text = response.text

        # 3. Extract text content
        content = trafilatura.extract(
            html_text,
            include_links=True,
            include_tables=True,
        )

        if not content:
            raise ValueError(f"No readable content found at URL: {target_url}")

        # 4. Extract rich metadata
        doc_metadata = trafilatura.extract_metadata(html_text)

        metadata = {
            "source": "url",
            "url": str(response.url),
            "status_code": response.status_code,
        }

        if doc_metadata:
            if doc_metadata.title:
                metadata["title"] = doc_metadata.title
            if doc_metadata.author:
                metadata["author"] = doc_metadata.author
            if doc_metadata.date:
                metadata["publish_date"] = doc_metadata.date
            if doc_metadata.url:
                metadata["canonical_url"] = doc_metadata.url

        page = ExtractedPage(
            page_number=1,
            content=content,
            content_type="text",
            metadata=metadata,
        )

        return [page]