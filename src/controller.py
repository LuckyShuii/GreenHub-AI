"""Controller layer bridging the API view and the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from configs import Settings, get_settings

from .materials import build_material, lookup_by_name
from .model import ClassificationResult, Model
from .regions import RegionCatalog, SearchResult

if TYPE_CHECKING:
    from .embedder import ImageEmbedder

from .schemas import (
    ManualConfirmResponse,
    ManualMaterialRequest,
    MatchResponse,
    NameCandidateResponse,
    UnknownMaterialError,
)


class Controller:
    """Coordinate model access and manual catalog searches."""

    model: Model

    def __init__(
        self,
        embedder: ImageEmbedder | None = None,
        catalog: RegionCatalog | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize model, region catalog and matching thresholds."""
        self.model = Model(embedder)
        self.catalog = catalog
        self.settings = settings or get_settings()

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        """Set the shared image embedder on the model."""
        self.model.set_embedder(embedder)

    async def get_model_response(self, image: Image.Image, region: str) -> ClassificationResult:
        """Run the image classifier; return a match or candidate list."""
        return await self.model.classify(image, region)

    async def classify(self, image: Image.Image, region: str) -> ClassificationResult:
        """Alias of :meth:`get_model_response` with the public name."""
        return await self.get_model_response(image, region)

    async def resolve_manual(self, request: ManualMaterialRequest) -> ManualConfirmResponse:
        """Resolve a manual material name through the region catalog."""
        query = request.nom
        if request.slug is not None:
            slug = lookup_by_name(request.slug)
            if slug is not None:
                material = build_material(slug)
                return MatchResponse(
                    material_name=query,
                    bin_color=material.column,
                    accepted_materials=[material],
                    query=query,
                    match_type="exact",
                    score=1.0,
                )
        if self.catalog is None:
            raise RuntimeError("Region catalog is not initialized.")
        settings = self.settings
        result: SearchResult = self.catalog.search(
            query,
            request.region,
            accept_threshold=settings.match_accept_threshold,
            candidate_threshold=settings.match_candidate_threshold,
            ambiguity_margin=settings.match_ambiguity_margin,
            max_candidates=settings.match_max_candidates,
            min_query_length=settings.match_min_query_length,
        )
        if result.status == "matched" and result.entry is not None:
            return MatchResponse(
                material_name=result.entry.name,
                bin_color=result.entry.bin_color,
                accepted_materials=list(result.entry.accepted_materials) if result.entry.accepted_materials else None,
                query=query,
                match_type=result.match_type,
                score=result.score,
            )
        if result.status == "needs_choice":
            return NameCandidateResponse(query=query, candidates=list(result.candidates))

        fallback = lookup_by_name(query)
        if fallback is not None:
            material = build_material(fallback)
            return MatchResponse(
                material_name=query,
                bin_color=material.column,
                accepted_materials=[material],
                query=query,
                match_type="exact",
                score=1.0,
            )
        raise UnknownMaterialError(query=query, region=request.region, suggestions=list(result.candidates))
