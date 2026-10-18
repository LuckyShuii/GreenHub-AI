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

    async def point_exists(self, collection_name: str, point_id: int | str) -> bool:
        """Return False only for a genuine absence; re-raise other errors."""
        try:
            records = await self._client.retrieve(
                collection_name=collection_name, ids=[point_id]
            )
        except Exception as error:  # noqa: BLE001
            if getattr(error, "status_code", None) == 404:
                return False
            logger.exception(
                "Existence check failed in '%s' for ID %s.", collection_name, point_id
            )
            raise
        return len(records) > 0


    async def update_payload(
        self, collection_name: str, point_ids: list[int | str], payload: QdrantPoint
    ) -> bool:
        """Refresh metadata for existing points without re-embedding images."""
        try:
            await self._client.set_payload(
                collection_name=collection_name,
                payload=payload.payload.model_dump(),
                points=point_ids,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Payload refresh failed in '%s' for IDs %s.", collection_name, point_ids)
            raise
        return True


    async def upsert(self, collection_name: str, point: QdrantPoint) -> bool:
        """Insert or replace a point in a region collection."""
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
        except Exception:  # noqa: BLE001
            logger.exception("Upsert failed in '%s' for ID %s.", collection_name, point.point_id)
            raise
        logger.debug("Upserted point %s into '%s'.", point.point_id, collection_name)
        return True
