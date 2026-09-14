"""Unit and white-box tests for the model layer."""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from configs import get_settings
from src.model import (
    FR_BIN_COLOR_CODE,
    Model,
    Response,
    UnknownMaterialError,
)


class TestResponse:
    """Unit tests for the Response factory."""

    def test_from_material_maps_known_material(self) -> None:
        """A known material yields the correct bin color."""
        response = Response.from_material("glass")
        assert response.material_name == "glass"
        assert response.bin_color == FR_BIN_COLOR_CODE["glass"]

    @pytest.mark.parametrize("material", list(FR_BIN_COLOR_CODE.keys()))
    def test_from_material_covers_all_known_materials(
        self, material: str
    ) -> None:
        """Every mapped material builds a valid Response (branch coverage)."""
        response = Response.from_material(material)
        assert response.bin_color == FR_BIN_COLOR_CODE[material]

    def test_from_material_raises_on_unknown(self) -> None:
        """An unmapped material triggers UnknownMaterialError."""
        with pytest.raises(UnknownMaterialError) as exc:
            Response.from_material("unobtanium")
        assert "unobtanium" in str(exc.value)


class TestModel:
    """White-box tests for the Model wrapper internals."""

    @pytest.fixture
    def model(self, monkeypatch: pytest.MonkeyPatch) -> Model:
        """Return a Model whose pipeline is mocked to avoid downloads."""
        monkeypatch.setenv("PIPELINE_MODEL_NAME", "test/model")
        get_settings.cache_clear()
        fake = MagicMock(return_value=[{"label": "Metal", "score": 0.9}])
        monkeypatch.setattr(
            "src.model.transformers.pipeline",
            lambda *args, **kwargs: fake,
        )
        return Model()

    def test_run_inference_returns_lowercase_label(
        self, model: Model, sample_image: Image.Image
    ) -> None:
        """_run_inference lowercases the top predicted label."""
        assert model._run_inference(sample_image) == "metal"

    @pytest.mark.asyncio
    async def test_predict_material_offloads_and_maps(
        self, model: Model, sample_image: Image.Image
    ) -> None:
        """predict_material returns a mapped Response asynchronously."""
        response = await model.predict_material(sample_image)
        assert isinstance(response, Response)
        assert response.material_name == "metal"

    @pytest.mark.asyncio
    async def test_predict_material_propagates_unknown(
        self, model: Model, sample_image: Image.Image
    ) -> None:
        """An unknown label propagates UnknownMaterialError through async."""
        model.model.return_value = [{"label": "Wood", "score": 0.8}]
        with pytest.raises(UnknownMaterialError):
            await model.predict_material(sample_image)
