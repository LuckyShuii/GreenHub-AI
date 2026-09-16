"""Unit tests for the Qdrant vector repository."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from qdrant_client.http import models as qmodels

from src.repository import VectorRepository
from src.schemas import QdrantPoint

COLLECTION: str = "ile-de-france"


@pytest.fixture
def client() -> AsyncMock:
    """Build an asynchronous Qdrant client mock.

    Returns:
        An AsyncMock exposing the client surface used by the repository.
    """
    mock = AsyncMock()
    mock.collection_exists = AsyncMock(return_value=False)
    mock.create_collection = AsyncMock()
    mock.upsert = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def repository(client: AsyncMock) -> VectorRepository:
    """Build a repository bound to the mocked client.

    Args:
        client: Mocked Qdrant client.

    Returns:
        A VectorRepository instance.
    """
    return VectorRepository(client)


class TestEnsureCollection:
    """Collection provisioning behaviour."""

    @pytest.mark.asyncio
    async def test_creates_missing_collection(
        self, repository: VectorRepository, client: AsyncMock
    ) -> None:
        """A missing collection is created with cosine distance.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
        """
        await repository.ensure_collection(COLLECTION, 384)

        client.create_collection.assert_awaited_once()
        kwargs = client.create_collection.await_args.kwargs
        assert kwargs["collection_name"] == COLLECTION
        assert kwargs["vectors_config"].size == 384
        assert kwargs["vectors_config"].distance == qmodels.Distance.COSINE

    @pytest.mark.asyncio
    async def test_skips_existing_collection(
        self, repository: VectorRepository, client: AsyncMock
    ) -> None:
        """An existing collection is left untouched.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
        """
        client.collection_exists.return_value = True

        await repository.ensure_collection(COLLECTION, 384)

        client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_is_idempotent_across_calls(
        self, repository: VectorRepository, client: AsyncMock
    ) -> None:
        """Repeated calls create the collection at most once.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
        """
        await repository.ensure_collection(COLLECTION, 384)
        client.collection_exists.return_value = True
        await repository.ensure_collection(COLLECTION, 384)

        assert client.create_collection.await_count == 1


class TestUpsert:
    """Point insertion behaviour."""

    @pytest.mark.asyncio
    async def test_inserts_point_with_payload(
        self,
        repository: VectorRepository,
        client: AsyncMock,
        sample_point: QdrantPoint,
    ) -> None:
        """The point id, vector and payload reach the client.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
            sample_point: Point fixture.
        """
        result = await repository.upsert(COLLECTION, sample_point)

        assert result is True
        client.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_client_error(
        self,
        repository: VectorRepository,
        client: AsyncMock,
        sample_point: QdrantPoint,
    ) -> None:
        """Client failures are reported as a falsy result.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
            sample_point: Point fixture.
        """
        client.upsert.side_effect = RuntimeError("connection reset")

        result = await repository.upsert(COLLECTION, sample_point)

        assert result is False


class TestPointExists:
    """Existence check behaviour."""

    @pytest.mark.asyncio
    async def test_returns_true_when_point_present(
        self, repository: VectorRepository, client: AsyncMock
    ) -> None:
        """A non-empty retrieve result means the point exists.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
        """
        client.retrieve.return_value = [object()]

        assert await repository.point_exists(COLLECTION, 0) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_point_absent(
        self, repository: VectorRepository, client: AsyncMock
    ) -> None:
        """An empty retrieve result means the point is missing.

        Args:
            repository: Repository under test.
            client: Mocked Qdrant client.
        """
        client.retrieve.return_value = []

        assert await repository.point_exists(COLLECTION, 0) is False