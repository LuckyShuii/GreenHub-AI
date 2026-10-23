"""Controller layer bridging the API view and the model."""

from __future__ import annotations

from PIL import Image

from .embedder import ImageEmbedder
from .materials import build_material, lookup_by_name
from .model import ClassificationResult, Model
from .regions import lookup_by_name as region_lookup_by_name
from .schemas import (
    AcceptedMaterial,
    ManualMaterialRequest,
    MatchResponse,
    UnknownMaterialError,
)


class Controller:
    """Coordinates model access for the viewer layer."""

    model: Model

    def __init__(self, embedder: ImageEmbedder | None = None) -> None:
        self.model = Model(embedder)

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        self.model.set_embedder(embedder)

    async def get_model_response(
        self, image: Image.Image, region: str
    ) -> ClassificationResult:
        """Run the image classifier; returns a match or candidate list."""
        return await self.model.classify(image, region)

    async def classify(
        self, image: Image.Image, region: str
    ) -> ClassificationResult:
        """Alias of :meth:`get_model_response` with the public name."""
        return await self.get_model_response(image, region)

    async def resolve_manual(
        self, request: ManualMaterialRequest
    ) -> MatchResponse:
        """Resolve a manual user selection to a MatchResponse.

        Looks up the local region catalog by slug or name (case/accent
        insensitive) and raises :class:`UnknownMaterialError` if absent.
        """
        region = request.region
        slug_or_name = request.slug or request.nom or ""

        # 1. Try slug lookup first (covers both slug and label inputs).
        slug = lookup_by_name(slug_or_name)
        if slug is not None:
            material = build_material(slug)
            return MatchResponse(
                material_name=slug_or_name,
                bin_color=material.column,
                accepted_materials=[material],
            )

        # 2. Try region-specific lookup (case/accent insensitive on name).
        entry = region_lookup_by_name(region, slug_or_name)
        if entry is not None:
            accepted_raw = entry.get("accepted_materials")
            accepted: list[AcceptedMaterial] | None = None
            if isinstance(accepted_raw, list) and accepted_raw:
                try:
                    accepted = [
                        AcceptedMaterial.model_validate(item)
                        for item in accepted_raw
                        if isinstance(item, dict)
                    ]
                except Exception:  # noqa: BLE001
                    accepted = None
            return MatchResponse(
                material_name=str(entry.get("nom", slug_or_name)),
                bin_color=str(entry.get("poubelle", "unknown")),
                accepted_materials=accepted,
            )

        raise UnknownMaterialError(query=slug_or_name, region=region)
