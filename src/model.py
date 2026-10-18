"""Waste classification model using Qdrant vector similarity search."""

from __future__ import annotations

from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from logging_config import get_logger

from .embedder import ImageEmbedder
from .schemas import AcceptedMaterial, MaterialMetadata

logger = get_logger(__name__)


class UnknownRegionError(Exception):
    """Raised when the requested region has no Qdrant collection."""

    def __init__(self, region: str) -> None:
        """Store the unknown region name."""
        self.region = region
        super().__init__(f"Unknown region: {region}")


class NoMatchError(Exception):
    """Raised when no similar reference vector is found in Qdrant."""

    def __init__(self, region: str) -> None:
        """Store the region in which no match was found."""
        self.region = region
        super().__init__(f"No match found in region: {region}")


class Response(BaseModel):
    """Public prediction response without the internal similarity score."""

    material_name: str
    bin_color: str
    accepted_materials: list[AcceptedMaterial] | None = None


def parse_accepted_materials(
    payload: dict[str, Any],
) -> list[AcceptedMaterial] | None:
    """Validate stored materials, returning None when absent or invalid."""
    try:
        metadata = MaterialMetadata.model_validate(
            {"accepted_materials": payload.get("accepted_materials")}
        )
    except ValidationError as error:
        logger.warning("Invalid accepted_materials in payload: %s", error)
        return None
    return metadata.accepted_materials


class Model:
    """Asynchronous waste classifier backed by Qdrant similarity search."""

    def __init__(self, embedder: ImageEmbedder | None = None) -> None:
        settings = get_settings()
        self._embedder = embedder
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        """Inject the application-wide image embedder."""
        self._embedder = embedder

    async def predict_material(self, image: Image.Image, region: str) -> Response:
        if not await self._client.collection_exists(region):
            raise UnknownRegionError(region)
        if self._embedder is None:
            raise RuntimeError("Image embedder has not been initialized.")

        vector = await self._embedder.embed(image)
        results = await self._client.query_points(
            collection_name=region, query=vector, limit=1, with_payload=True
        )
        if not results.points:
            raise NoMatchError(region)

        payload: dict[str, Any] = results.points[0].payload or {}
        return Response(
            material_name=str(payload.get("nom", "unknown")),
            bin_color=str(payload.get("poubelle", "unknown")),
            accepted_materials=parse_accepted_materials(payload),
        )
