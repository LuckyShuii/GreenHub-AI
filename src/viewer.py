"""FastAPI application that indexes region collections and serves the Greener API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import Settings, get_settings
from logging_config import configure_logging, get_logger

from .controller import Controller
from .fetcher import ImageFetcher
from .model import ClassificationResult, NoMatchError
from .model import UnknownRegionError as ModelUnknownRegionError
from .regions import (
    RegionCatalog,
    discover_region_files,
    load_region_payloads,
)
from .regions import (
    UnknownRegionError as CatalogUnknownRegionError,
)
from .repository import VectorRepository
from .schemas import ManualConfirmResponse, ManualMaterialRequest, UnknownMaterialError

if TYPE_CHECKING:
    from .embedder import ImageEmbedder

logger = get_logger("greener.viewer")


class Viewer(FastAPI):
    """Greener FastAPI application handling region indexing and HTTP routes."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the application, its lifespan and its routes.

        Args:
            settings: Application settings. Defaults to the cached environment settings.
        """
        self.settings: Settings = settings or get_settings()
        self.controller: Controller | None = None
        self.region_catalog: RegionCatalog | None = None
        super().__init__(title="Greener Waste Classifier", lifespan=self._lifespan)
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        """Index missing regions and build the controller before serving requests."""
        configure_logging(self.settings.log_level)
        from .embedder import ImageEmbedder

        embedder: ImageEmbedder = ImageEmbedder(
            model_path=self.settings.embedding_model_path,
            device=self.settings.device,
        )
        client: AsyncQdrantClient = AsyncQdrantClient(host=self.settings.qdrant_host, port=self.settings.qdrant_port)
        try:
            await self._index_missing_regions(client, embedder)
        finally:
            await client.close()
        self.region_catalog = RegionCatalog.from_directory(Path(self.settings.data_dir))
        self.controller = Controller(embedder, self.region_catalog, self.settings)
        logger.info("Greener API ready.")
        yield

    async def _index_missing_regions(self, client: AsyncQdrantClient, embedder: ImageEmbedder) -> None:
        """Upload images into Qdrant for every region without a collection.

        Args:
            client: Asynchronous Qdrant client.
            embedder: Shared image embedder.
        """
        region_files: list[Path] = discover_region_files(Path(self.settings.data_dir))
        missing: list[Path] = [
            path for path in region_files if not await client.collection_exists(collection_name=path.stem)
        ]
        if not missing:
            logger.info("All %d region collection(s) already exist.", len(region_files))
            return

        from .indexer import RegionIndexer

        indexer: RegionIndexer = RegionIndexer(
            embedder=embedder,
            fetcher=ImageFetcher(
                root_dir=Path(self.settings.image_dir),
                semaphore=asyncio.Semaphore(self.settings.max_concurrent_uploads),
            ),
            repository=VectorRepository(client),
            images_per_label=self.settings.images_per_label,
        )
        for region_file in missing:
            logger.info("Indexing region '%s'.", region_file.stem)
            await indexer.index_region(region_file.stem, load_region_payloads(region_file))

    def _register_routes(self) -> None:
        """Attach the HTTP endpoints to the application."""
        self.add_api_route(
            "/greener/upload/dechets",
            self.upload_dechets,
            methods=["POST"],
            response_model=None,
        )
        self.add_api_route(
            "/greener/upload/dechets/confirm",
            self.confirm_dechets,
            methods=["POST"],
            response_model=ManualConfirmResponse,
            responses={404: {"description": "Unknown region or no matching material."}},
        )

    def _get_controller(self) -> Controller:
        """Return the controller or raise 503 if startup is not complete."""
        if self.controller is None:
            raise HTTPException(status_code=503, detail="Service is starting.")
        return self.controller

    @staticmethod
    def _decode_image(raw: bytes) -> Image.Image:
        """Decode raw bytes into a Pillow image or raise HTTP 400."""
        try:
            image: Image.Image = Image.open(BytesIO(raw))
            image.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
        return image

    async def upload_dechets(
        self,
        file: UploadFile = File(...),
        region: str = Query("ile_de_france", min_length=1),
    ) -> ClassificationResult:
        """Classify an uploaded waste image for the given region."""
        controller: Controller = self._get_controller()
        raw: bytes = await file.read()
        if len(raw) > self.settings.max_upload_size:
            raise HTTPException(status_code=413, detail="Uploaded file is too large.")
        image: Image.Image = self._decode_image(raw)
        try:
            return await controller.classify(image, region)
        except ModelUnknownRegionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NoMatchError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def confirm_dechets(self, request: ManualMaterialRequest) -> ManualConfirmResponse:
        """Confirm a manual user selection from a previous needs_choice response."""
        controller: Controller = self._get_controller()
        try:
            return await controller.resolve_manual(request)
        except CatalogUnknownRegionError as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": str(exc), "query": request.nom, "region": exc.region, "suggestions": []},
            ) from exc
        except UnknownMaterialError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": str(exc),
                    "query": exc.query,
                    "region": exc.region,
                    "suggestions": [item.model_dump() for item in exc.suggestions],
                },
            ) from exc
