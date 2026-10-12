"""Pydantic models for waste metadata and vector indexing."""

from pydantic import BaseModel, Field


class MaterialMetadata(BaseModel):
    """Catalog metadata attached to a waste item."""

    collection_material_code: str | None = None
    collection_material_name: str | None = None
    collection_materials: list[str] = Field(default_factory=list)


class WasteItem(MaterialMetadata):
    """Represent a waste entry loaded from a region JSON file."""

    nom: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)


class QdrantPayload(MaterialMetadata):
    """Metadata stored alongside each Qdrant vector."""

    nom: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    poubelle: str = Field(..., min_length=1)


class QdrantPoint(BaseModel):
    """Fully structured point ready for Qdrant insertion."""

    point_id: int = Field(..., ge=0)
    vector: list[float]
    payload: QdrantPayload
