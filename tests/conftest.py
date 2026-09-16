"""Shared fixtures and helpers for the waste classification test suite."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.schemas import QdrantPayload, QdrantPoint

EMBEDDING_DIM: int = 384
TEST_IMAGE_NAME: str = "test.jpeg"


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    """Return the embedding dimension used across the test suite.

    Returns:
        The vector dimensionality expected by fake embedders.
    """
    return EMBEDDING_DIM


@pytest.fixture
def sample_image() -> Image.Image:
    """Build a deterministic in-memory RGB image.

    Returns:
        A small solid-color PIL image suitable for embedding stubs.
    """
    return Image.new("RGB", (64, 64), color=(12, 200, 87))


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Serialize the sample image to PNG bytes.

    Args:
        sample_image: Source PIL image.

    Returns:
        PNG-encoded bytes of the sample image.
    """
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Return a byte payload that cannot be decoded as an image.

    Returns:
        Arbitrary non-image bytes.
    """
    return b"this-is-definitely-not-an-image"


@pytest.fixture
def sample_payload() -> QdrantPayload:
    """Build a valid Qdrant payload.

    Returns:
        A payload describing a glass bottle in Ile-de-France.
    """
    return QdrantPayload(
        nom="bouteille en verre",
        region="ile-de-france",
        poubelle="verte",
    )


@pytest.fixture
def sample_vector(embedding_dimension: int) -> list[float]:
    """Build a deterministic unit-norm embedding vector.

    Args:
        embedding_dimension: Target vector dimensionality.

    Returns:
        A normalized list of floats.
    """
    rng = np.random.default_rng(seed=42)
    vector = rng.normal(size=embedding_dimension)
    vector /= np.linalg.norm(vector)
    return vector.tolist()


@pytest.fixture
def sample_point(
    sample_vector: list[float], sample_payload: QdrantPayload
) -> QdrantPoint:
    """Build a fully structured Qdrant point.

    Args:
        sample_vector: Embedding vector for the point.
        sample_payload: Metadata attached to the point.

    Returns:
        A QdrantPoint ready for insertion.
    """
    return QdrantPoint(
        point_id=0, vector=sample_vector, payload=sample_payload
    )


@pytest.fixture
def region_file(tmp_path: Path) -> Path:
    """Write a valid region JSON file into a temporary data directory.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the created region JSON file.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload: list[dict[str, str]] = [
        {"nom": "bouteille en verre", "poubelle": "verte"},
        {"nom": "carton", "poubelle": "jaune"},
        {"nom": "reste alimentaire", "poubelle": "marron"},
    ]
    path = data_dir / "ile-de-france.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def malformed_region_file(tmp_path: Path) -> Path:
    """Write a region JSON file containing partially invalid entries.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the created region JSON file.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    content: list[Any] = [
        {"nom": "carton", "poubelle": "jaune"},
        {"nom": "", "poubelle": "jaune"},
        {"nom": "plastique"},
        {"poubelle": "verte"},
    ]
    path = data_dir / "bretagne.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


@pytest.fixture
def test_image_path() -> Path:
    """Locate the shared test image used by performance tests.

    Returns:
        Path to data/test.png.

    Raises:
        FileNotFoundError: If the image is missing from the repository.
    """
    path = Path(__file__).parent / TEST_IMAGE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Required test asset not found: {path.resolve()}"
        )
    return path


class FakeEmbedder:
    """Deterministic embedder replacement avoiding model downloads.

    Attributes:
        dimension: Dimensionality of the produced vectors.
        call_count: Number of embed invocations performed.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        """Initialize the fake embedder.

        Args:
            dimension: Dimensionality of generated vectors.
        """
        self.dimension = dimension
        self.call_count = 0

    async def embed(self, image: Image.Image) -> list[float]:
        """Return a deterministic vector derived from image content.

        Args:
            image: PIL image to encode.

        Returns:
            A normalized embedding vector.
        """
        self.call_count += 1
        seed = sum(image.convert("RGB").resize((4, 4)).tobytes()[:16])
        rng = np.random.default_rng(seed=seed)
        vector = rng.normal(size=self.dimension)
        vector /= np.linalg.norm(vector)
        return vector.tolist()


class FailingEmbedder(FakeEmbedder):
    """Embedder stub that always raises to exercise error handling."""

    async def embed(self, image: Image.Image) -> list[float]:
        """Raise unconditionally.

        Args:
            image: Ignored PIL image.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("embedding backend unavailable")


class FakeFetcher:
    """Image fetcher stub returning a fixed number of images.

    Attributes:
        requested: Recorded (label, count) pairs for assertions.
    """

    def __init__(self, image: Image.Image, available: int = 3) -> None:
        """Initialize the fake fetcher.

        Args:
            image: Image returned for every request.
            available: Number of images returned per call.
        """
        self._image = image
        self._available = available
        self.requested: list[tuple[str, int]] = []

    async def fetch(
        self, client: Any, label: str, count: int
    ) -> list[Image.Image]:
        """Return a fixed list of images.

        Args:
            client: Unused HTTP client placeholder.
            label: Waste label being searched.
            count: Number of images requested.

        Returns:
            A list of identical PIL images.
        """
        self.requested.append((label, count))
        return [self._image] * min(count, self._available)


@pytest.fixture
def fake_embedder(embedding_dimension: int) -> FakeEmbedder:
    """Provide a deterministic embedder instance.

    Args:
        embedding_dimension: Vector dimensionality.

    Returns:
        A FakeEmbedder ready for injection.
    """
    return FakeEmbedder(embedding_dimension)


@pytest.fixture
def fake_fetcher(sample_image: Image.Image) -> FakeFetcher:
    """Provide a fetcher stub returning the sample image.

    Args:
        sample_image: Image returned by the stub.

    Returns:
        A FakeFetcher instance.
    """
    return FakeFetcher(sample_image)


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Provide a session-scoped event loop for async fixtures.

    Yields:
        The event loop used by session-scoped async fixtures.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
