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
from .regions import normalize_region_key


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

    def __init__(self) -> None:
        """Create the embedder and the Qdrant client from settings."""
        settings = get_settings()
        self._embedder = ImageEmbedder(settings.embedding_model_path, settings.device)
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )

    async def _resolve_collection(self, region: str) -> str:
        """Map a user-supplied region to the name of an existing collection.

        The exact name is tried first; otherwise every existing collection is
        compared using the canonical region key.

        Raises:
            UnknownRegionError: If no collection matches the region.
        """
        if await self._client.collection_exists(collection_name=region):
            return region
        target: str = normalize_region_key(region)
        if target:
            response = await self._client.get_collections()
            for collection in response.collections:
                if normalize_region_key(collection.name) == target:
                    return collection.name
        raise UnknownRegionError(region)

    async def predict_material(
        self, image: Image.Image, region: str) -> MatchResponse | CandidateResponse:
        """Classify a waste image against the collection of the given region."""
        collection_name: str = await self._resolve_collection(region)
        if self._embedder is None:
            msg = "Embedder has not been attached to the Model instance."
            raise RuntimeError(msg)

        settings = get_settings()
        vector = await self._embedder.embed(image)
        results = await self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=settings.top_k,
            with_payload=True,
        )
        if not results.points:
            raise NoMatchError(region)