"""Waste classification model using Qdrant vector similarity search."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import pydantic
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from src.embedder import ImageEmbedder


class UnknownRegionError(Exception):
    """Raised when the requested region has no Qdrant collection."""

    def __init__(self, region: str) -> None:
        """Initialize the error with the offending region name.

        Args:
            region: The region name that has no matching collection.

        """
        self.region = region
        super().__init__(f"Unknown region: {region}")


class NoMatchError(Exception):
    """Raised when no similar reference vector is found in Qdrant."""

    def __init__(self, region: str) -> None:
        """Initialize the error with the searched region.

        Args:
            region: The region collection that returned no match.

        """
        self.region = region
        super().__init__(f"No match found in region: {region}")


class Response(pydantic.BaseModel):
    """Structured prediction response returned by the model.

    Attributes:
        material_name: The name of the detected waste item.
        bin_color: The region-specific bin instruction.
        score: The similarity score of the closest match.

    """

    material_name: str
    bin_color: str
    score: float


class Model:
    """Asynchronous waste classifier backed by Qdrant similarity search.

    The image is embedded, then the region-specific collection is queried
    for the nearest reference vector. The associated payload carries the
    waste name and the region-specific bin color.
    """

    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor()

    def __init__(self) -> None:
        """Initialize the embedder and the async Qdrant client."""
        settings = get_settings()
        self._embedder = ImageEmbedder(settings.embedding_model_name,
                                       settings.device)
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    async def _embed_image(self, image: Image.Image) -> list[float]:
        """Embed an image synchronously in a worker thread.

        Args:
            image: The PIL image to embed.

        Returns:
            The embedding vector as a list of floats.

        """
        return await self._embedder.embed(image)

    async def _collection_exists(self, region: str) -> bool:
        """Check whether the region collection exists in Qdrant.

        Args:
            region: The region name used as the collection name.

        Returns:
            True if the collection exists, False otherwise.

        """
        return await self._client.collection_exists(region)

    async def predict_material(
        self, image: Image.Image, region: str
    ) -> Response:
        """Classify an image against a region-specific collection.

        Args:
            image: The PIL image to classify.
            region: The region whose collection is queried.

        Returns:
            A Response with the waste name, bin color, and score.

        Raises:
            UnknownRegionError: If the region collection does not exist.
            NoMatchError: If the search returns no result.

        """
        if not await self._collection_exists(region):
            raise UnknownRegionError(region)

        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(
            self._executor,
            self._embed_image,
            image,
        )

        results = await self._client.query_points(
            collection_name=region,
            query=await vector,
            limit=1,
            with_payload=True,
        )

        if not results:
            raise NoMatchError(region)

        best = results.points[0]
        payload = best.payload or {}
        return Response(
            material_name=str(payload.get("nom", "unknown")),
            bin_color=str(payload.get("poubelle", "unknown")),
            score=float(best.score),
        )