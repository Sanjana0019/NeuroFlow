from io import BytesIO
from pathlib import Path

from PIL import Image
import pytesseract

from backend.providers.base import ChatMessage
from backend.providers.client import get_client
from backend.providers.router import RoutingCriteria

from pipelines.ingestion.models import ExtractedPage


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 1024


class ImageExtractor:
    """Extract OCR text and generate a vision-LLM description from images."""

    async def extract(self, file_path: str | Path) -> list[ExtractedPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: {extension}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        with Image.open(path) as image:
            image = image.convert("RGB")

            original_width, original_height = image.size

            # Resize only when the longest side exceeds 1024 pixels.
            image.thumbnail(
                (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
                Image.Resampling.LANCZOS,
            )

            resized_width, resized_height = image.size

            # OCR the resized image.
            ocr_text = pytesseract.image_to_string(
                image,
                config="--psm 6",
            ).strip()

            # Encode the resized image for the vision model.
            image_buffer = BytesIO()

            image.save(
                image_buffer,
                format="JPEG",
                quality=85,
            )

            image_bytes = image_buffer.getvalue()

        # Convert image bytes to a data URL so the provider abstraction
        # can send the image as multimodal content.
        import base64

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        image_data_url = (
            f"data:image/jpeg;base64,{image_base64}"
        )

        client = get_client()

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail for a document "
                            "ingestion pipeline. Identify the main subject, "
                            "layout, objects, diagrams, charts, tables, "
                            "relationships, and any visually important "
                            "information. Do not invent information that "
                            "cannot be seen."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            )
        ]

        routing_criteria = RoutingCriteria(
            task_type="image_description",
            require_vision=True,
        )

        result = await client.chat(
            messages,
            routing_criteria,
        )

        description = result.content.strip()

        if ocr_text:
            content = (
                f"{description}\n\n"
                f"Text found in image: {ocr_text}"
            )
        else:
            content = description

        return [
            ExtractedPage(
                page_number=1,
                content=content,
                content_type="image_description",
                metadata={
                    "source": "image",
                    "filename": path.name,
                    "original_width": original_width,
                    "original_height": original_height,
                    "width": resized_width,
                    "height": resized_height,
                    "ocr": bool(ocr_text),
                    "vision_model": result.model,
                },
            )
        ]