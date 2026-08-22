import base64

import pytest

from pipelines.ingestion.extractors.image_extractor import ImageExtractor
from pipelines.ingestion.models import ExtractedPage


class FakeGenerationResult:
    content = "A detailed description of the NeuroFlow test image."
    model = "fake-vision-model"


class FakeClient:
    def __init__(self):
        self.messages = None
        self.routing_criteria = None

    async def chat(self, messages, routing_criteria, **kwargs):
        self.messages = messages
        self.routing_criteria = routing_criteria

        return FakeGenerationResult()


@pytest.mark.asyncio
async def test_image_extractor(
    monkeypatch,
    tmp_path,
):
    from PIL import Image, ImageDraw, ImageFont

    image_path = tmp_path / "test.png"

    # Create a large image with clearly readable text.
    image = Image.new(
        "RGB",
        (1600, 900),
        "white",
    )

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf",
        72,
    )

    draw.text(
        (100, 100),
        "NeuroFlow Image Test",
        fill="black",
        font=font,
    )

    draw.text(
        (100, 220),
        "Task 4 Multimodal Ingestion",
        fill="black",
        font=font,
    )

    image.save(image_path)

    # Replace the real NeuroFlow client with a fake client.
    fake_client = FakeClient()

    monkeypatch.setattr(
        "pipelines.ingestion.extractors.image_extractor.get_client",
        lambda: fake_client,
    )

    extractor = ImageExtractor()

    pages = await extractor.extract(image_path)

    # Exactly one ExtractedPage should be returned.
    assert len(pages) == 1

    page = pages[0]

    assert isinstance(page, ExtractedPage)

    # Image extractor should create image_description content.
    assert page.page_number == 1
    assert page.content_type == "image_description"

    # Vision model description should be included.
    assert (
        "A detailed description of the NeuroFlow test image."
        in page.content
    )

    # OCR text should also be included.
    assert "Text found in image:" in page.content

    # Metadata checks.
    assert page.metadata["source"] == "image"

    assert page.metadata["original_width"] == 1600
    assert page.metadata["original_height"] == 900

    # Longest side must be <= 1024.
    assert page.metadata["width"] <= 1024
    assert page.metadata["height"] <= 1024

    assert page.metadata["ocr"] is True

    assert page.metadata["vision_model"] == "fake-vision-model"

    # Verify that vision routing was requested.
    assert (
        fake_client.routing_criteria.task_type
        == "image_description"
    )

    assert (
        fake_client.routing_criteria.require_vision
        is True
    )

    # Verify that a multimodal message was created.
    assert fake_client.messages is not None
    assert len(fake_client.messages) == 1

    message = fake_client.messages[0]

    assert message.role == "user"

    assert isinstance(message.content, list)

    assert message.content[0]["type"] == "text"

    assert message.content[1]["type"] == "image_url"

    # Verify that the image was converted to a base64 data URL.
    image_url = message.content[1]["image_url"]["url"]

    assert image_url.startswith(
        "data:image/jpeg;base64,"
    )

    encoded_image = image_url.split(",", 1)[1]

    decoded_image = base64.b64decode(encoded_image)

    assert len(decoded_image) > 0