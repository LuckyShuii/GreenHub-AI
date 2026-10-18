"""Pydantic models for waste metadata and vector indexing."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MaterialSlug = Literal[
    "verre",
    "tissus-vetements",
    "plastique",
    "bouchons-plastique",
    "bouchons-liege",
    "papier-emballages",
    "livres",
    "CD-DVD",
    "piles",
    "chaussures",
    "metaux",
    "telephones",
    "dechets-verts",
    "ampoules",
    "cartouches",
    "materiel-informatique",
    "jouets",
]

MATERIAL_CATALOG: dict[str, tuple[str, str]] = {
    "verre": ("VER", "Verre"),
    "tissus-vetements": ("VET", "Tissus / Vêtements"),
    "plastique": ("PLST", "Plastique"),
    "bouchons-plastique": ("BCHP", "Bouchons plastique"),
    "bouchons-liege": ("BCHL", "Bouchons liège"),
    "papier-emballages": ("PPR", "Papier / Emballages"),
    "livres": ("LIV", "Livres"),
    "CD-DVD": ("CD", "CD / DVD"),
    "piles": ("PIL", "Piles"),
    "chaussures": ("CHS", "Chaussures"),
    "metaux": ("MET", "Métaux"),
    "telephones": ("TLF", "Téléphones"),
    "dechets-verts": ("GEEN", "Déchets verts"),
    "ampoules": ("AMPL", "Ampoules"),
    "cartouches": ("CER", "Cartouches"),
    "materiel-informatique": ("INF", "Matériel informatique"),
    "jouets": ("PLS", "Jouets"),
}


class AcceptedMaterial(BaseModel):
    """A collection material accepted for a waste item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: MaterialSlug
    column: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_catalog_consistency(self) -> Self:
        """Ensure column and label match the catalog entry for the slug."""
        expected = MATERIAL_CATALOG[self.slug]
        if (self.column, self.label) != expected:
            msg = (
                f"Inconsistent material '{self.slug}': expected "
                f"{expected}, got {(self.column, self.label)}."
            )
            raise ValueError(msg)
        return self


class MaterialMetadata(BaseModel):
    """Catalog metadata attached to a waste item."""

    accepted_materials: list[AcceptedMaterial] | None = None

    @field_validator("accepted_materials")
    @classmethod
    def normalize_empty(
        cls, value: list[AcceptedMaterial] | None
    ) -> list[AcceptedMaterial] | None:
        """Convert an empty list to None and reject duplicate slugs."""
        if not value:
            return None
        slugs = [material.slug for material in value]
        if len(slugs) != len(set(slugs)):
            msg = f"Duplicate material slugs: {slugs}."
            raise ValueError(msg)
        return value


class WasteItem(MaterialMetadata):
    """Represent a waste entry loaded from a region JSON file."""

    model_config = ConfigDict(extra="forbid")

    nom: str = Field(..., min_length=1)
    poubelle: str = Field(..., min_length=1)


class QdrantPayload(MaterialMetadata):
    """Metadata stored alongside each Qdrant vector."""

    nom: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    poubelle: str = Field(..., min_length=1)


class QdrantPoint(BaseModel):
    """Fully structured point ready for Qdrant insertion."""

    point_id: int | str
    vector: list[float]
    payload: QdrantPayload
