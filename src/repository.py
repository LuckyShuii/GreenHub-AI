"""Asynchronous Qdrant repository supporting multiple region collections."""

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from logging_config import get_logger
from .schemas import QdrantPoint

logger = get_logger(__name__)


class VectorRepository:
    """Manage multiple Qdrant collections, one per region."""

    def __init__(self, client: AsyncQdrantClient) -> None:
        """Initialize the repository.

        Args:
            client: Asynchronous Qdrant client instance.

        """
        self._client = client

    async def ensure_collection(
        self, collection_name: str, dimension: int
    ) -> None:
        """Create a region collection if it does not already exist.

        Args:
            collection_name: Name of the region collection.
            dimension: Dimensionality of the stored vectors.

        """
        exists = await self._client.collection_exists(collection_name)
        if exists:
            logger.info("Collection '%s' already exists.", collection_name)
            return

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("Created collection '%s'.", collection_name)

    async def point_exists(self, collection_name: str, point_id: int) -> bool:
        """Check whether a point exists in a region collection.

        Args:
            collection_name: Name of the region collection.
            point_id: Identifier of the point to look up.

        Returns:
            True if the point exists, False otherwise.

        """
        try:
            records = await self._client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
            )
            return len(records) > 0
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Existence check failed in '%s' for ID %d: %s",
                collection_name,
                point_id,
                error,
            )
            return False

    async def upsert(self, collection_name: str, point: QdrantPoint) -> bool:
        """Insert a point into a region collection if absent.

        Args:
            collection_name: Name of the region collection.
            point: Fully structured point to insert.

        Returns:
            True if the point was inserted, False otherwise.

        """
        try:
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point.point_id,
                        vector=point.vector,
                        payload=point.payload.model_dump(),
                    )
                ],
            )
            logger.debug(
                "Upserted point %d into '%s'.", point.point_id, collection_name
            )
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Upsert failed in '%s' for ID %d: %s",
                collection_name,
                point.point_id,
                error,
            )
            return False
