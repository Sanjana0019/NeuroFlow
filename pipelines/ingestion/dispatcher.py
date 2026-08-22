from pathlib import Path
from urllib.parse import urlparse

from pipelines.ingestion.extractors.csv_extractor import CSVExtractor
from pipelines.ingestion.extractors.docx_extractor import DOCXExtractor
from pipelines.ingestion.extractors.image_extractor import ImageExtractor
from pipelines.ingestion.extractors.pdf_extractor import PDFExtractor
from pipelines.ingestion.extractors.pptx_extractor import PPTXExtractor
from pipelines.ingestion.extractors.text_extractor import TextExtractor
from pipelines.ingestion.extractors.url_extractor import URLExtractor
from pipelines.ingestion.models import ExtractedPage


class IngestionDispatcher:
    """Select the appropriate extractor for an ingestion source."""

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    }

    FILE_EXTRACTORS = {
        ".pdf": PDFExtractor,
        ".docx": DOCXExtractor,
        ".csv": CSVExtractor,
        ".txt": TextExtractor,
        ".pptx": PPTXExtractor,
    }

    def dispatch(self, source: str) -> list[ExtractedPage]:
        """Extract content from a URL or local file."""

        if not source or not source.strip():
            raise ValueError("Ingestion source cannot be empty")

        source = source.strip()

        if self._is_url(source):
            return URLExtractor().extract(source)

        path = Path(source)
        extension = path.suffix.lower()

        if extension in self.IMAGE_EXTENSIONS:
            return ImageExtractor().extract(path)

        extractor_class = self.FILE_EXTRACTORS.get(extension)

        if extractor_class is None:
            raise ValueError(
                f"Unsupported ingestion source type: {extension or 'unknown'}"
            )

        return extractor_class().extract(path)

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)

        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)