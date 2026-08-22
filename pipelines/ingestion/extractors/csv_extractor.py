from pathlib import Path

import pandas as pd

from pipelines.ingestion.models import ExtractedPage


class CSVExtractor:
    """Extract structured tabular data from CSV files."""

    def extract(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if not path.is_file():
            raise ValueError(f"CSV path is not a file: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc

        content = self._dataframe_to_markdown(df)

        metadata = {
            "source": "csv",
            "filename": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
        }

        page = ExtractedPage(
            page_number=1,
            content=content,
            content_type="table",
            metadata=metadata,
        )

        return [page]

    @staticmethod
    def _dataframe_to_markdown(df: pd.DataFrame) -> str:
        """Convert a DataFrame to Markdown without extra dependencies."""

        headers = [str(column) for column in df.columns]

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]

        for _, row in df.iterrows():
            values = [str(value) for value in row]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)