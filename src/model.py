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
from .schemas import (
    AcceptedMaterial,
    CandidateResponse,
    MatchResponse,
    MaterialCandidate,
    MaterialMetadata,
)

logger = get_logger(__name__)


class UnknownRegionError(Exception):
    """Raised when the requested region has no Qdrant collection."""

    def __init__(self, region: str) -> None:
        """Store the unknown region name."""
        self.region = region
        super().__init__(f"Unknown region: {region}")


class NoMatchError(Exception):
    """Raised when Qdrant returns no candidate at all."""

    def __init__(self, region: str) -> None:
        """Store the region in which no match was found."""
        self.region = region
        super().__init__(f"No match found in region: {region}")


# Public classification result. The internal similarity score never leaks.
ClassificationResult = MatchResponse | CandidateResponse


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
        settings = get_settings()
        self._embedder = embedder
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        """Inject the application-wide image embedder."""
        self._embedder = embedder

    async def classify(self, image: Image.Image, region: str) -> ClassificationResult:
        """Classify a waste image, returning a match or candidate list.

        The similarity score is read from ``point.score`` and compared to
        ``settings.similarity_threshold``. It is never propagated to the
        response object.
        """
        if not await self._client.collection_exists(region):
            raise UnknownRegionError(region)
        if self._embedder is None:
            raise RuntimeError("Image embedder has not been initialized.")

        settings = get_settings()
        threshold = settings.similarity_threshold
        top_k = settings.top_k

        vector = await self._embedder.embed(image)
        results = await self._client.query_points(
            collection_name=region,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        if not results.points:
            raise NoMatchError(region)

        # Sort by score descending so the first point is the best match.
        scored_points = sorted(
            results.points,
            key=lambda p: getattr(p, "score", 0.0) or 0.0,
            reverse=True,
        )

        best = scored_points[0]
        best_score: float = getattr(best, "score", 0.0) or 0.0
        best_payload: dict[str, Any] = best.payload or {}

        if best_score >= threshold:
            return MatchResponse(
                material_name=str(best_payload.get("nom", "unknown")),
                bin_color=str(best_payload.get("poubelle", "unknown")),
                accepted_materials=parse_accepted_materials(best_payload),
            )

        candidates = deduplicate_candidates(
            candidate_from_payload(point.payload or {}) for point in scored_points
        )
        logger.info(
            "Below threshold (%.3f < %.3f) for region '%s': %d candidate(s).",
            best_score,
            threshold,
            region,
            len(candidates),
        )
        return CandidateResponse(candidates=candidates)


class Response(BaseModel):
    """Backward-compatible alias for the matched response."""

    material_name: str
    bin_color: str
    accepted_materials: list[AcceptedMaterial] | None = None


# Re-export for backwards compatibility with existing imports.
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
