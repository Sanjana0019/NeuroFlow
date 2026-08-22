from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
import pytesseract

from pipelines.ingestion.models import ExtractedPage


OCR_THRESHOLD = 50


class PDFExtractor:
    """Extract text, OCR scanned pages, and tables from PDF files."""

    async def extract(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        pages: list[ExtractedPage] = []

        pdf = pdfium.PdfDocument(str(path))

        try:
            with pdfplumber.open(str(path)) as plumber_pdf:
                for page_index, pdfium_page in enumerate(pdf):
                    page_number = page_index + 1

                    # First try normal digital-PDF text extraction.
                    text_page = pdfium_page.get_textpage()
                    text = text_page.get_text_range().strip()

                    # If very little text was extracted, treat the page
                    # as potentially scanned and run OCR.
                    was_ocr = len(text) < OCR_THRESHOLD

                    if was_ocr:
                        text = self._ocr_page(pdfium_page)

                    if text:
                        pages.append(
                            ExtractedPage(
                                page_number=page_number,
                                content=text,
                                content_type="text",
                                metadata={
                                    "page": page_number,
                                    "ocr": was_ocr,
                                },
                            )
                        )

                    # Extract tables separately using pdfplumber.
                    plumber_page = plumber_pdf.pages[page_index]
                    tables = plumber_page.extract_tables()

                    for table_index, table in enumerate(tables, start=1):
                        markdown = self._table_to_markdown(table)

                        if markdown:
                            pages.append(
                                ExtractedPage(
                                    page_number=page_number,
                                    content=markdown,
                                    content_type="table",
                                    metadata={
                                        "page": page_number,
                                        "table_index": table_index,
                                    },
                                )
                            )
        finally:
            pdf.close()

        return pages

    @staticmethod
    def _ocr_page(pdf_page) -> str:
        """Rasterize a PDF page and run Tesseract OCR."""

        bitmap = pdf_page.render(scale=2)
        image = bitmap.to_pil()

        try:
            return pytesseract.image_to_string(
                image,
                config="--psm 6",
            ).strip()
        finally:
            image.close()

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """Convert a pdfplumber table into a Markdown table."""

        if not table:
            return ""

        cleaned_rows = []

        for row in table:
            cleaned_rows.append(
                [
                    (cell or "")
                    .replace("|", "\\|")
                    .replace("\n", " ")
                    .strip()
                    for cell in row
                ]
            )

        if not cleaned_rows:
            return ""

        column_count = max(len(row) for row in cleaned_rows)

        # Make every row the same width.
        for row in cleaned_rows:
            row.extend([""] * (column_count - len(row)))

        header = cleaned_rows[0]
        separator = ["---"] * column_count

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]

        for row in cleaned_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)