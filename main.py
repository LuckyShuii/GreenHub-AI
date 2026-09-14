"""Application entrypoint exposing the waste classification API."""

import io

import uvicorn
from fastapi import File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from configs import get_settings
from src.model import UnknownRegionError, NoMatchError
from src.viewer import Viewer

servapp = Viewer()


@servapp.post("/greener/upload/dechets")
async def upload_file(
    file: UploadFile = File(...),
    region: str = Form(...),
) -> JSONResponse:
    """Classify an uploaded waste image for a given region.

    Args:
        file: The uploaded image file.
        region: The user's region, matching a Qdrant collection name.

    Returns:
        A JSON response with the material name and bin color.

    Raises:
        HTTPException: If the file is not a valid image (400), if the
            material cannot be classified (422), or if the region is
            unknown (404).

    """
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from exc

    try:
        response = await servapp.get_response(image, region)
    except UnknownRegionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NoMatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return JSONResponse(response.model_dump())


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app=servapp, host=settings.ai_host, port=settings.ai_port)
