"""Waste classification model using Qdrant vector similarity search."""

from __future__ import annotations

from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient

from configs import get_settings
from logging_config import get_logger

from .embedder import ImageEmbedder
from .materials import candidate_from_payload, deduplicate_candidates
from .regions import normalize_region_key
from .schemas import (
    AcceptedMaterial,
    CandidateResponse,
    MatchResponse,
    MaterialCandidate,
    MaterialMetadata,
)

logger = get_logger(__name__)

ClassificationResult = MatchResponse | CandidateResponse


class UnknownRegionError(Exception):
    """Raised when the requested region has no Qdrant collection."""

    def __init__(self, region: str) -> None:
        """Store the unknown region name."""
        self.region: str = region
        super().__init__(f"Unknown region: {region}")


class NoMatchError(Exception):
    """Raised when Qdrant returns no candidate at all."""

    def __init__(self, region: str) -> None:
        """Store the region in which no match was found."""
        self.region: str = region
        super().__init__(f"No match found in region: {region}")


def parse_accepted_materials(
    payload: dict[str, Any],
) -> list[AcceptedMaterial] | None:
    """Validate stored materials, returning None when absent or invalid."""
    try:
        metadata = MaterialMetadata.model_validate(
            {"accepted_materials": payload.get("accepted_materials")}
        )
    except ValidationError as error:
        logger.warning("Invalid accepted_materials in payload: %s", error)
        return None
    return metadata.accepted_materials


class Model:
    """Asynchronous waste classifier backed by Qdrant similarity search."""

    def __init__(self, embedder: ImageEmbedder | None = None) -> None:
        """Create the Qdrant client and store the optional embedder."""
        settings = get_settings()
        self._embedder: ImageEmbedder | None = embedder
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        """Inject the application-wide image embedder."""
        self._embedder = embedder

    async def _resolve_collection(self, region: str) -> str:
        """Return the existing Qdrant collection name matching a region.

        The literal name is tried first, then every existing collection is
        compared using the canonical region key.

        Raises:
            UnknownRegionError: If no collection matches the region.
        """
        if await self._client.collection_exists(collection_name=region):
            return region

        target: str = normalize_region_key(region)
        if target:
            response = await self._client.get_collections()
            for collection in response.collections:
                if normalize_region_key(collection.name) == target:
                    return collection.name
        raise UnknownRegionError(region)

    async def classify(self, image: Image.Image, region: str) -> ClassificationResult:
        """Classify a waste image, returning a match or a candidate list.

        The similarity score is compared to the configured threshold and is
        never exposed in the response.
        """
        collection_name: str = await self._resolve_collection(region)
        if self._embedder is None:
            msg = "Image embedder has not been initialized."
            raise RuntimeError(msg)

        settings = get_settings()
        threshold: float = settings.similarity_threshold

        vector = await self._embedder.embed(image)
        results = await self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=settings.top_k,
            with_payload=True,
        )
        if not results.points:
            raise NoMatchError(region)

        scored_points = sorted(
            results.points,
            key=lambda point: point.score or 0.0,
            reverse=True,
        )
        best = scored_points[0]
        best_score: float = best.score or 0.0
        best_payload: dict[str, Any] = best.payload or {}

        if best_score >= threshold:
            return MatchResponse(
                material_name=str(best_payload.get("nom", "unknown")),
                bin_color=str(best_payload.get("poubelle", "unknown")),
                accepted_materials=parse_accepted_materials(best_payload),
            )

        raw_candidates: list[MaterialCandidate] = [
            candidate_from_payload(point.payload or {}) for point in scored_points
        ]
        candidates: list[MaterialCandidate] = deduplicate_candidates(raw_candidates)
        logger.info(
            "Below threshold (%.3f < %.3f) for region '%s': %d candidate(s).",
            best_score,
            threshold,
            region,
            len(candidates),
        )
        return CandidateResponse(candidates=candidates)


class Response(BaseModel):
    """Backward-compatible model for a matched response."""

    material_name: str
    bin_color: str
    accepted_materials: list[AcceptedMaterial] | None = None


__all__ = [
    "CandidateResponse",
    "ClassificationResult",
    "MatchResponse",
    "MaterialCandidate",
    "Model",
    "NoMatchError",
    "Response",
    "UnknownRegionError",
]