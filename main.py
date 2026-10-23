"""FastAPI entry point exposing the Greener classification API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from configs import get_settings
from logging_config import configure_logging, get_logger
from src.model import NoMatchError, UnknownRegionError
from src.schemas import (
    ManualMaterialRequest,
    MatchResponse,
    UnknownMaterialError,
)
from src.viewer import Viewer
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and prepare the shared viewer at startup."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.viewer = Viewer()
    app.state.logger = get_logger("greener.main")
    app.state.logger.info("Greener API ready.")
    yield


app = FastAPI(title="Greener Waste Classifier", lifespan=lifespan)


def _decode_image(raw: bytes) -> Image.Image:
    """Decode raw bytes into a Pillow image or raise HTTPException(400)."""
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@app.post(
    "/greener/upload/dechets",
    response_model=None,
)
async def upload_dechets(
    file: UploadFile = File(...),
    region: str = Query("ile_de_france", min_length=1),
):
    """Classify an uploaded waste image for the given region."""
    viewer: Viewer = app.state.viewer
    raw = await file.read()
    image = _decode_image(raw)
    try:
        result = await viewer.controller.classify(image, region)
    except UnknownRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoMatchError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post(
    "/greener/upload/dechets/confirm",
    response_model=MatchResponse,
)
async def confirm_dechets(request: ManualMaterialRequest) -> MatchResponse:
    """Confirm a manual user selection from a previous needs_choice response."""
    viewer: Viewer = app.state.viewer
    try:
        return await viewer.confirm(request)
    except UnknownMaterialError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app=app, host=settings.ai_host, port=settings.ai_port)