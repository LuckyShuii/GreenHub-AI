"""FastAPI application that indexes region collections and serves the Greener API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from qdrant_client import AsyncQdrantClient

from configs import Settings, get_settings
from logging_config import configure_logging, get_logger

from .controller import Controller
from .fetcher import ImageFetcher, to_rgb
from .model import ClassificationResult, NoMatchError
from .model import UnknownRegionError as ModelUnknownRegionError
from .regions import (
    RegionCatalog,
    discover_region_files,
    load_region_payloads,
    normalize_region_key,
)
from .regions import (
    UnknownRegionError as CatalogUnknownRegionError,
)
from .repository import VectorRepository
from .schemas import ManualConfirmResponse, ManualMaterialRequest, UnknownMaterialError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .embedder import ImageEmbedder

logger = get_logger("greener.viewer")

UPLOAD_PATH: str = "/greener/upload/dechets"
ALLOWED_IMAGE_FORMATS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
UPLOAD_READ_CHUNK: int = 1024 * 1024
# Room for multipart boundaries and part headers on top of the file itself.
MULTIPART_OVERHEAD: int = 64 * 1024
DOCS_PATHS: frozenset[str] = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
API_CSP: str = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
UI_CSP: str = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


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
        self._inference_slots: asyncio.Semaphore = asyncio.Semaphore(self.settings.max_concurrent_uploads)
        docs_enabled: bool = self.settings.enable_docs
        super().__init__(
            title="Greener Waste Classifier",
            lifespan=self._lifespan,
            docs_url="/docs" if docs_enabled else None,
            redoc_url="/redoc" if docs_enabled else None,
            openapi_url="/openapi.json" if docs_enabled else None,
        )
        self.middleware("http")(self._security_middleware)
        self._register_routes()
        self._register_test_ui()

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        """Index missing regions and build the controller before serving requests."""
        configure_logging(self.settings.log_level)
        from .embedder import ImageEmbedder

        embedder: ImageEmbedder = ImageEmbedder(
            model_path=self.settings.embedding_model_path,
            device=self.settings.device,
            semaphore=asyncio.Semaphore(self.settings.max_concurrent_uploads),
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
            UPLOAD_PATH,
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
        self.add_api_route("/greener/regions", self.list_regions, methods=["GET"])

    def _register_test_ui(self) -> None:
        """Serve the optional, git-ignored test UI at ``/greener/`` when it is present."""
        ui_dir: Path = Path(self.settings.test_ui_dir)
        if not (ui_dir / "index.html").is_file():
            return
        index_file: Path = ui_dir / "index.html"

        async def redirect_to_ui() -> RedirectResponse:
            # Relative target so it keeps working behind a reverse-proxy prefix.
            return RedirectResponse(url="greener/", status_code=307)

        async def serve_ui() -> FileResponse:
            return FileResponse(index_file, media_type="text/html")

        self.add_api_route("/greener", redirect_to_ui, methods=["GET"], include_in_schema=False)
        self.add_api_route("/greener/", serve_ui, methods=["GET"], include_in_schema=False)
        self.mount("/greener/test-ui", StaticFiles(directory=ui_dir), name="test-ui")
        logger.info("Test UI enabled from '%s'.", ui_dir)

    async def _security_middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Reject oversized uploads early and add security headers to every response."""
        if request.method == "POST" and request.url.path == UPLOAD_PATH:
            content_length: str | None = request.headers.get("content-length")
            if content_length is not None:
                if not content_length.isdigit():
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})
                if int(content_length) > self.settings.max_upload_size + MULTIPART_OVERHEAD:
                    return JSONResponse(status_code=413, content={"detail": "Uploaded file is too large."})

        response: Response = await call_next(request)
        path: str = request.url.path
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if path not in DOCS_PATHS:
            is_ui: bool = path == "/greener/" or path.startswith("/greener/test-ui/")
            headers.setdefault("Content-Security-Policy", UI_CSP if is_ui else API_CSP)
        return response

    async def list_regions(self) -> dict[str, list[str]]:
        """Return the identifiers of the loaded regions."""
        if self.region_catalog is None:
            raise HTTPException(status_code=503, detail="Service is starting.")
        return {"regions": list(self.region_catalog.regions)}

    def _ensure_known_region(self, region: str) -> None:
        """Reject regions that are not part of the loaded catalog.

        This prevents the image endpoint from querying arbitrary Qdrant collections.
        """
        catalog: RegionCatalog | None = self.region_catalog
        if catalog is not None and normalize_region_key(region) not in catalog.regions:
            raise HTTPException(status_code=400, detail="Unknown region.")

    async def _read_upload(self, file: UploadFile) -> bytes:
        """Read an upload in chunks, aborting as soon as it exceeds the size limit."""
        limit: int = self.settings.max_upload_size
        buffer: bytearray = bytearray()
        while chunk := await file.read(UPLOAD_READ_CHUNK):
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
        if not buffer:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return bytes(buffer)

    def _get_controller(self) -> Controller:
        """Return the controller or raise 503 if startup is not complete."""
        if self.controller is None:
            raise HTTPException(status_code=503, detail="Service is starting.")
        return self.controller

    def _decode_image(self, raw: bytes) -> Image.Image:
        """Decode raw bytes into an RGB Pillow image or raise an HTTP error.

        Only common photo formats are accepted, and the pixel count is checked
        from the header before decoding to block decompression bombs.
        """
        try:
            image: Image.Image = Image.open(BytesIO(raw), formats=ALLOWED_IMAGE_FORMATS)
            width, height = image.size
            if width * height > self.settings.max_image_pixels:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            image.load()
            return to_rgb(image)
        except HTTPException:
            raise
        except Image.DecompressionBombError as exc:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.") from exc
        except Exception as exc:  # noqa: BLE001
            logger.info("Rejected undecodable upload: %s", type(exc).__name__)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or unsupported image. Accepted formats: {', '.join(ALLOWED_IMAGE_FORMATS)}.",
            ) from exc

    async def upload_dechets(
        self,
        file: UploadFile = File(...),
        region: str = Query("ile_de_france", min_length=1, max_length=64),
    ) -> ClassificationResult:
        """Classify an uploaded waste image for the given region."""
        controller: Controller = self._get_controller()
        self._ensure_known_region(region)
        raw: bytes = await self._read_upload(file)
        try:
            await asyncio.wait_for(self._inference_slots.acquire(), timeout=self.settings.inference_queue_timeout)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail="Server is busy, retry later.") from exc
        try:
            image: Image.Image = await asyncio.to_thread(self._decode_image, raw)
            return await controller.classify(image, region)
        except ModelUnknownRegionError as exc:
            raise HTTPException(status_code=400, detail="Unknown region.") from exc
        except NoMatchError as exc:
            raise HTTPException(status_code=404, detail="No match found.") from exc
        except ValueError as exc:
            logger.warning("Classification rejected: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid request.") from exc
        finally:
            self._inference_slots.release()

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
