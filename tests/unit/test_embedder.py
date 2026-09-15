"""Unit tests for the DINOv2 image embedder wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.embedder import ImageEmbedder

MODEL_NAME: str = "facebook/dinov2-small"
HIDDEN_SIZE: int = 384


def _build_model_stub(hidden_size: int = HIDDEN_SIZE) -> MagicMock:
    """Create a torch-free stand-in for a Hugging Face model.

    Args:
        hidden_size: Hidden dimensionality advertised by the config.

    Returns:
        A MagicMock mimicking the AutoModel interface.
    """
    model = MagicMock()
    model.config.hidden_size = hidden_size
    model.to.return_value = model
    outputs = MagicMock()
    outputs.last_hidden_state = torch.ones((1, 5, hidden_size))
    model.return_value = outputs
    return model


def _build_processor_stub() -> MagicMock:
    """Create a stand-in for a Hugging Face image processor.

    Returns:
        A MagicMock returning tensor-like batch encodings.
    """
    processor = MagicMock()
    encoding = MagicMock()
    encoding.to.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    encoding.__getitem__.side_effect = lambda key: torch.zeros(
        (1, 3, 224, 224)
    )
    processor.return_value = encoding
    return processor


@pytest.fixture
def patched_embedder() -> Any:
    """Instantiate an ImageEmbedder with mocked transformer components.

    Yields:
        A tuple of (embedder, model stub, processor stub).
    """
    model = _build_model_stub()
    processor = _build_processor_stub()
    with (
        patch(
            "src.embedder.AutoModel.from_pretrained", return_value=model
        ) as model_loader,
        patch(
            "src.embedder.AutoImageProcessor.from_pretrained",
            return_value=processor,
        ) as processor_loader,
    ):
        embedder = ImageEmbedder(MODEL_NAME, "cpu")
        yield embedder, model, processor, model_loader, processor_loader


class TestImageEmbedderInitialization:
    """Construction behaviour of the embedder."""

    def test_loads_model_and_processor(self, patched_embedder: Any) -> None:
        """Both the processor and model are loaded from the given name.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
        """
        _, _, _, model_loader, processor_loader = patched_embedder
        model_loader.assert_called_once_with(MODEL_NAME)
        processor_loader.assert_called_once_with(MODEL_NAME)

    def test_exposes_model_hidden_size(self, patched_embedder: Any) -> None:
        """The dimension attribute mirrors the model hidden size.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
        """
        embedder, _, _, _, _ = patched_embedder
        assert embedder.dimension == HIDDEN_SIZE

    def test_switches_model_to_eval_mode(
        self, patched_embedder: Any
    ) -> None:
        """Inference mode is enabled at construction time.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
        """
        _, model, _, _, _ = patched_embedder
        model.eval.assert_called_once()


class TestImageEmbedderInference:
    """Embedding behaviour of the embedder."""

    @pytest.mark.asyncio
    async def test_embed_returns_expected_dimension(
        self, patched_embedder: Any, sample_image: Image.Image
    ) -> None:
        """The async embed call returns a vector of the right size.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
            sample_image: In-memory PIL image.
        """
        embedder, _, _, _, _ = patched_embedder
        vector = await embedder.embed(sample_image)

        assert isinstance(vector, list)
        assert len(vector) == HIDDEN_SIZE
        assert all(isinstance(value, float) for value in vector)

    @pytest.mark.asyncio
    async def test_embed_does_not_block_event_loop(
        self, patched_embedder: Any, sample_image: Image.Image
    ) -> None:
        """Concurrent embeddings complete without deadlocking.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
            sample_image: In-memory PIL image.
        """
        import asyncio

        embedder, _, _, _, _ = patched_embedder
        vectors = await asyncio.gather(
            *(embedder.embed(sample_image) for _ in range(8))
        )

        assert len(vectors) == 8
        assert all(len(vector) == HIDDEN_SIZE for vector in vectors)

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(
        self, patched_embedder: Any, sample_image: Image.Image
    ) -> None:
        """The same image yields identical vectors across calls.

        Args:
            patched_embedder: Fixture bundling the embedder and stubs.
            sample_image: In-memory PIL image.
        """
        embedder, _, _, _, _ = patched_embedder
        first = await embedder.embed(sample_image)
        second = await embedder.embed(sample_image)

        assert np.allclose(first, second)