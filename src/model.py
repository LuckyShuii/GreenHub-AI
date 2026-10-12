"""Waste classification model using Qdrant vector similarity search."""

from __future__ import annotations

from typing import Any

import pydantic
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from .embedder import ImageEmbedder


class UnknownRegionError(Exception):
    """Raised when the requested region has no Qdrant collection."""

    def __init__(self, region: str) -> None:
        self.region = region
        super().__init__(f"Unknown region: {region}")


class NoMatchError(Exception):
    """Raised when no similar reference vector is found in Qdrant."""

    def __init__(self, region: str) -> None:
        self.region = region
        super().__init__(f"No match found in region: {region}")


class Response(pydantic.BaseModel):
    """Public prediction response without the internal similarity score."""

    material_name: str
    bin_color: str
    collection_material_code: str | None = None
    collection_material_name: str | None = None
    collection_materials: list[str] = pydantic.Field(default_factory=list)


class Model:
    """Asynchronous waste classifier backed by Qdrant similarity search."""

    def __init__(self) -> None:
        settings = get_settings()
        self._embedder = ImageEmbedder(settings.embedding_model_path, settings.device)
        self._client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    async def predict_material(self, image: Image.Image, region: str) -> Response:
        """Classify an image and return the stored waste metadata."""
        if not await self._client.collection_exists(region):
            raise UnknownRegionError(region)
        vector = await self._embedder.embed(image)
        results: Any = await self._client.query_points(
            collection_name=region,
            query=vector,
            limit=1,
            with_payload=True,
        )
        if not results.points:
            raise NoMatchError(region)
        payload = results.points[0].payload or {}
        return Response(
            material_name=str(payload.get("nom", "unknown")),
            bin_color=str(payload.get("poubelle", "unknown")),
            collection_material_code=payload.get("collection_material_code"),
            collection_material_name=payload.get("collection_material_name"),
            collection_materials=list(payload.get("collection_materials") or []),
        )
