"""Pydantic models for the multi-region waste indexing pipeline."""

from pydantic import BaseModel, Field


class WasteItem(BaseModel):
    """Represent a single waste entry loaded from a region JSON file.

    Attributes:
        nom: Human-readable name of the waste item.
        region: Name of the region the item belongs to.

    """

    nom: str = Field(..., min_length=1, description="Waste item name.")
    region: str = Field(..., min_length=1, description="Region name.")


class QdrantPayload(BaseModel):
    """Structured payload stored alongside each Qdrant vector.

    Attributes:
        nom: Human-readable name of the waste item.
        region: Region associated with the collection.
        poubelle: Target bin color for this item in this region.

    """

    nom: str = Field(..., min_length=1, description="Waste item name.")
    region: str = Field(..., min_length=1, description="Region name.")
    poubelle: str = Field(..., min_length=1, description="Target bin.")


class QdrantPoint(BaseModel):
    """Fully structured point ready for Qdrant insertion.

    Attributes:
        point_id: Deterministic identifier derived from the JSON index.
        vector: Embedding vector representing the reference image.
        payload: Metadata associated with the point.

    """

    point_id: int = Field(..., ge=0, description="Deterministic point ID.")
    vector: list[float] = Field(..., description="Embedding vector.")
    payload: QdrantPayload = Field(..., description="Associated metadata.")
