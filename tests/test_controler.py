"""Unit tests for the controller layer with a mocked model."""

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.Controller import Controller
from src.model import Response


@pytest.mark.asyncio
async def test_Controller_delegates_to_model(
    monkeypatch: pytest.MonkeyPatch, sample_image: Image.Image
) -> None:
    """The controller forwards the image and returns the model response."""
    expected = Response(material_name="glass", bin_color="Poubelle VERTE")
    monkeypatch.setattr(
        "src.Controller.Model.__init__", lambda self: None
    )
    Controller = Controller()
    Controller.model = AsyncMock()
    Controller.model.predict_material.return_value = expected

    result = await Controller.get_model_response(sample_image)

    Controller.model.predict_material.assert_awaited_once_with(sample_image)
    assert result == expected
