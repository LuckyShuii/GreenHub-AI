"""Black-box tests exercising the HTTP API end to end."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main
from src.model import Response, UnknownMaterialError


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with the controller response stubbed."""
    stub = AsyncMock(
        return_value=Response(
            material_name="glass",
            bin_color="Poubelle VERTE (verre)",
        )
    )
    monkeypatch.setattr(main.servapp.Controller, "get_model_response", stub)
    return TestClient(main.servapp)


def test_upload_valid_image_returns_bin_color(
    client: TestClient, image_bytes: bytes
) -> None:
    """A valid image upload returns HTTP 200 and the mapped bin color."""
    response = client.post(
        "/greener/upload/dechets",
        files={"file": ("waste.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["material_name"] == "glass"
    assert body["bin_color"] == "Poubelle VERTE (verre)"


def test_upload_invalid_image_returns_400(client: TestClient) -> None:
    """A non-image payload is rejected with HTTP 400."""
    response = client.post(
        "/greener/upload/dechets",
        files={"file": ("bad.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_unknown_material_returns_422(
    monkeypatch: pytest.MonkeyPatch, image_bytes: bytes
) -> None:
    """An unmapped material surfaces as HTTP 422."""
    stub = AsyncMock(side_effect=UnknownMaterialError("wood"))
    monkeypatch.setattr(main.servapp.Controller, "get_model_response", stub)
    client = TestClient(main.servapp)

    response = client.post(
        "/greener/upload/dechets",
        files={"file": ("waste.png", image_bytes, "image/png")},
    )
    assert response.status_code == 422
