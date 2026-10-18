"""Startup indexing integration for the FastAPI viewer."""

import asyncio
from collections.abc import AsyncIterator

from fastapi import FastAPI
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from logging_config import configure_logging, get_logger

from .controller import Controller
from .embedder import ImageEmbedder
from .fetcher import ImageFetcher
from .indexer import RegionIndexer
from .model import Response
from .regions import discover_region_files, load_region_payloads
from .repository import VectorRepository

logger = get_logger(__name__)


async def run_startup_indexing(embedder: ImageEmbedder | None = None) -> None:
    """Discover regions and index each into its own Qdrant collection."""
    settings = get_settings()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    if embedder is None:
        embedder = ImageEmbedder(
            settings.embedding_model_path,
            settings.device,
            asyncio.Semaphore(settings.max_concurrent_uploads),
        )

    repository = VectorRepository(client)
    fetcher = ImageFetcher(
        semaphore=asyncio.Semaphore(settings.max_concurrent_uploads),
        root_dir=settings.image_dir,
    )
    indexer = RegionIndexer(embedder, fetcher, repository, settings.images_per_label)

    try:
        region_files = discover_region_files(settings.data_dir)
    except FileNotFoundError as error:
        logger.error("Cannot start indexing: %s", error)
        await client.close()
        return

    try:
        for json_path in region_files:
            collection_name = json_path.stem
            try:
                payloads = load_region_payloads(json_path)
            except ValueError as error:
                logger.error("Skipping region '%s': %s", collection_name, error)
                continue
            await indexer.index_region(collection_name, payloads)
    finally:
        await client.close()

    logger.info("Startup indexing finished for all regions.")


async def lifespan(app: "Viewer") -> AsyncIterator[None]:
    """Load the model once, share it, then run indexing."""
    settings = get_settings()
    configure_logging(settings.log_level)

    embedder = ImageEmbedder(
        settings.embedding_model_path,
        settings.device,
        asyncio.Semaphore(settings.max_concurrent_uploads),
    )
    app.controller.set_embedder(embedder)

    try:
        await run_startup_indexing(embedder)
    except Exception as error:  # noqa: BLE001
        logger.exception("Unexpected error during startup indexing: %s", error)

    yield
    logger.info("Server shutdown.")


class Viewer(FastAPI):
    """FastAPI application holding a controller instance."""

    controller: Controller

    def __init__(self, **kwargs: object) -> None:
        super().__init__(lifespan=lifespan, **kwargs)  # type: ignore
        self.controller = Controller()

    async def get_response(self, image: Image.Image, region: str) -> Response:
        return await self.controller.get_model_response(image, region)
