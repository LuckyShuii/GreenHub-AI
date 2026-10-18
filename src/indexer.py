"""Multi-region indexing orchestration."""

import asyncio
import uuid

from logging_config import get_logger

from .embedder import ImageEmbedder
from .fetcher import ImageFetcher
from .repository import VectorRepository
from .schemas import QdrantPayload, QdrantPoint

logger = get_logger(__name__)

# Constant namespace: never change it, or every ID will change.
POINT_ID_NAMESPACE = uuid.UUID("3f9e5b3b-5b71-4a95-9d8b-0f5a4f2f6b9a")


class RegionIndexer:
    """Coordinate the indexing of every region collection."""

    def __init__(
        self,
        embedder: ImageEmbedder,
        fetcher: ImageFetcher,
        repository: VectorRepository,
        images_per_label: int,
    ) -> None:
        self._embedder = embedder
        self._fetcher = fetcher
        self._repository = repository
        self._images_per_label = images_per_label

    @staticmethod
    def _point_id(collection_name: str, label: str, image_index: int) -> str:
        """Return a deterministic UUID5 for one reference image."""
        identity = f"{collection_name}:{label}:{image_index}"
        return str(uuid.uuid5(POINT_ID_NAMESPACE, identity))

    async def _index_payload(self, collection_name: str, payload: QdrantPayload) -> int:
        point_ids = [
            self._point_id(collection_name, payload.nom, offset)
            for offset in range(self._images_per_label)
        ]

        if await self._repository.point_exists(collection_name, point_ids[0]):
            await self._repository.update_payload(
                collection_name,
                point_ids,
                QdrantPoint(point_id=point_ids[0], vector=[], payload=payload),
            )
            logger.info("Refreshed metadata for '%s' in '%s'.", payload.nom, collection_name)
            return 0

        images = await self._fetcher.fetch(payload.nom, self._images_per_label)

        inserted = 0
        for offset, image in enumerate(images):
            try:
                vector = await self._embedder.embed(image)
            except Exception as error:  # noqa: BLE001
                logger.warning("Embedding failed for '%s': %s", payload.nom, error)
                continue

            point = QdrantPoint(point_id=point_ids[offset], vector=vector, payload=payload)
            if await self._repository.upsert(collection_name, point):
                inserted += 1

        logger.info("Indexed %d vector(s) for '%s' in '%s'.", inserted, payload.nom, collection_name)
        return inserted

    async def index_region(self, collection_name: str, payloads: list[QdrantPayload]) -> None:
        await self._repository.ensure_collection(collection_name, self._embedder.dimension)
        results = await asyncio.gather(
            *(self._index_payload(collection_name, p) for p in payloads)
        )
        logger.info("Region '%s' complete. Total inserted: %d.", collection_name, sum(results))
