"""Viewer (presentation) layer wrapping the Controller for the FastAPI app."""

from __future__ import annotations

from io import BytesIO
from typing import Union

from PIL import Image

from .controller import Controller
from .schemas import (
    CandidateResponse,
    ManualMaterialRequest,
    MatchResponse,
)

Response = Union[MatchResponse, CandidateResponse]


class Viewer:
    """Thin async wrapper around :class:`Controller` for the HTTP layer."""

    def __init__(self, controller: Controller | None = None) -> None:
        self.controller = controller or Controller()

    async def get_response(self, image_bytes: bytes, region: str) -> Response:
        """Decode the uploaded image and ask the controller to classify it.

        Returns a :class:`MatchResponse` when the classifier is confident,
        or a :class:`CandidateResponse` when the user must disambiguate.
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid image payload.") from exc
        return await self.controller.classify(image, region)

    async def confirm(self, request: ManualMaterialRequest) -> MatchResponse:
        """Resolve a manual user selection to a confident :class:`MatchResponse`."""
        return await self.controller.resolve_manual(request)
