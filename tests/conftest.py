"""Shared pytest fixtures for the test suite."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import servapp


@pytest.fixture
def sample_image() -> Image.Image:
    """Return a small in-memory RGB image for inference tests."""
    return Image.new("RGB", (16, 16), color=(120, 120, 120))


@pytest.fixture
def image_bytes(sample_image: Image.Image) -> bytes:
    """Return the sample image encoded as PNG bytes for upload tests."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI TestClient bound to the application."""
    return TestClient(servapp)
