"""Startup indexing integration for the FastAPI viewer."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from .controller import Controller
from .embedder import ImageEmbedder
from .fetcher import ImageFetcher
from .indexer import RegionIndexer
from logging_config import configure_logging, get_logger
from .model import Response
from .regions import discover_region_files, load_region_payloads
from .repository import VectorRepository

logger = get_logger(__name__)


async def run_startup_indexing() -> None:
    """Discover regions and index each into its own Qdrant collection."""
    settings = get_settings()
    client = AsyncQdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port
    )
    embedder = ImageEmbedder(settings.embedding_model_name, settings.device)
    repository = VectorRepository(client)
    semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)
    fetcher = ImageFetcher(
        settings.request_timeout,
        settings.max_retries,
        semaphore,
        settings.save_images,
        settings.image_backup_dir,
    )
    indexer = RegionIndexer(
        embedder, fetcher, repository, settings.images_per_label
    )

    try:
        region_files = discover_region_files(settings.data_dir)
    except FileNotFoundError as error:
        logger.error("Cannot start indexing: %s", error)
        await client.close()
        return

    async with httpx.AsyncClient() as http_client:
        for json_path in region_files:
            collection_name = json_path.stem
            try:
                payloads = load_region_payloads(json_path)
            except ValueError as error:
                logger.error("Skipping region '%s': %s", collection_name, error)
                continue
            await indexer.index_region(collection_name, payloads, http_client)

    await client.close()
    logger.info("Startup indexing finished for all regions.")


@asynccontextmanager
async def lifespan(app: "Viewer") -> AsyncIterator[None]:
    """Run multi-region indexing at server startup.

    Args:
        app: The Viewer application instance.

    Yields:
        Control back to the application once indexing completes.

    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Server startup: launching multi-region indexing.")
    try:
        await run_startup_indexing()
    except Exception as error:  # noqa: BLE001
        logger.exception("Unexpected error during startup indexing: %s", error)
    yield
    logger.info("Server shutdown.")


class Viewer(FastAPI):
    """FastAPI application holding a controller instance."""

    controller: Controller

    def __init__(self, **kwargs: object) -> None:
        """Initialize the FastAPI app, controller and lifespan.

        Args:
            **kwargs: Keyword arguments forwarded to FastAPI.

        """
        super().__init__(lifespan=lifespan, **kwargs)  # type: ignore
        self.controller = Controller()

    async def get_response(self, image: Image.Image, region: str) -> Response:
        """Return the model response for an uploaded image.

        Args:
            image: The PIL image to classify.

        Returns:
            The classification Response.

        """
        return await self.controller.get_model_response(image, region)
