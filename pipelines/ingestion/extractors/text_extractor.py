from pathlib import Path

from pipelines.ingestion.models import ExtractedPage


class TextExtractor:
    """Extract text content from plain-text files."""

    def extract(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Text path is not a file: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Text file '{path}' is not valid UTF-8: {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Failed to read text file '{path}': {exc}"
            ) from exc

        if not content.strip():
            raise ValueError(f"Text file is empty: {path}")

        metadata = {
            "source": "text",
            "filename": path.name,
        }

        page = ExtractedPage(
            page_number=1,
            content=content,
            content_type="text",
            metadata=metadata,
        )

        return [page]