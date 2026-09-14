"""Multi-region indexing orchestration."""

import asyncio

import httpx

from .embedder import ImageEmbedder
from .fetcher import ImageFetcher
from logging_config import get_logger
from .repository import VectorRepository
from .schemas import QdrantPayload, QdrantPoint

logger = get_logger(__name__)


class RegionIndexer:
    """Coordinate the indexing of every region collection."""

    def __init__(
        self,
        embedder: ImageEmbedder,
        fetcher: ImageFetcher,
        repository: VectorRepository,
        images_per_label: int,
    ) -> None:
        """Initialize the region indexer.

        Args:
            embedder: Image embedder instance.
            fetcher: Image fetcher instance.
            repository: Multi-collection Qdrant repository.
            images_per_label: Number of images fetched per label.

        """
        self._embedder = embedder
        self._fetcher = fetcher
        self._repository = repository
        self._images_per_label = images_per_label

    async def _index_payload(
        self,
        collection_name: str,
        index: int,
        payload: QdrantPayload,
        client: httpx.AsyncClient,
    ) -> int:
        """Index all images for a single waste payload.

        Args:
            collection_name: Target region collection.
            index: Positional index used for deterministic IDs.
            payload: Waste payload to index.
            client: Shared asynchronous HTTP client.

        Returns:
            The number of vectors inserted for this payload.

        """
        base_id = index * self._images_per_label
        if await self._repository.point_exists(collection_name, base_id):
            logger.info(
                "Item '%s' already indexed in '%s'. Skipping.",
                payload.nom,
                collection_name,
            )
            return 0

        logger.info(
            "Fetching images for '%s' in region '%s'.",
            payload.nom,
            collection_name,
        )
        images = await self._fetcher.fetch(
            client, payload.nom, self._images_per_label
        )

        inserted = 0
        for offset, image in enumerate(images):
            try:
                vector = await self._embedder.embed(image)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Embedding failed for '%s': %s", payload.nom, error
                )
                continue

            point = QdrantPoint(
                point_id=base_id + offset,
                vector=vector,
                payload=payload,
            )
            if await self._repository.upsert(collection_name, point):
                inserted += 1

        logger.info(
            "Indexed %d vector(s) for '%s' in '%s'.",
            inserted,
            payload.nom,
            collection_name,
        )
        return inserted

    async def index_region(
        self,
        collection_name: str,
        payloads: list[QdrantPayload],
        client: httpx.AsyncClient,
    ) -> None:
        """Index every payload of a region concurrently.

        Args:
            collection_name: Target region collection.
            payloads: List of payloads for the region.
            client: Shared asynchronous HTTP client.

        """
        await self._repository.ensure_collection(
            collection_name, self._embedder.dimension
        )
        tasks = [
            self._index_payload(collection_name, index, payload, client)
            for index, payload in enumerate(payloads)
        ]
        results = await asyncio.gather(*tasks)
        logger.info(
            "Region '%s' complete. Total inserted: %d.",
            collection_name,
            sum(results),
        )
