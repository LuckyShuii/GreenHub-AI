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

# Slug -> display label (used for case-insensitive lookup by name).
MATERIAL_LABELS: dict[str, str] = {slug: label for slug, (_, label) in MATERIAL_CATALOG.items()}


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


# --- Public API response schemas (no `score` field) ---


class MaterialCandidate(BaseModel):
    """A single waste candidate returned when classification is uncertain.

    The similarity score is intentionally NOT exposed: it is used internally
    by the model layer only.
    """

    model_config = ConfigDict(extra="forbid")

    nom: str
    poubelle: str
    slug: MaterialSlug | None = None
    label: str | None = None
    accepted_materials: list[AcceptedMaterial] | None = None


class MatchResponse(BaseModel):
    """Response returned when classification is confident enough."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["matched"] = "matched"
    material_name: str
    bin_color: str
    accepted_materials: list[AcceptedMaterial] | None = None


class CandidateResponse(BaseModel):
    """Response returned when classification is below the confidence threshold."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_choice"] = "needs_choice"
    requires_user_choice: bool = True
    candidates: list[MaterialCandidate]


class ManualMaterialRequest(BaseModel):
    """Body of the manual confirm endpoint.

    The user supplies either a known slug or a free-text name. The region
    is required to validate the answer against the local catalog.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str | None = Field(default=None, min_length=1)
    nom: str | None = Field(default=None, min_length=1)
    region: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def ensure_one_identifier(self) -> Self:
        """Make sure exactly one of slug or nom is provided."""
        has_slug = self.slug is not None
        has_nom = self.nom is not None
        if has_slug == has_nom:
            msg = "Exactly one of 'slug' or 'nom' must be provided."
            raise ValueError(msg)
        return self


class UnknownMaterialError(Exception):
    """Raised when a manual lookup does not match any known material."""

    def __init__(self, query: str, region: str) -> None:
        """Store the offending query and region."""
        self.query = query
        self.region = region
        super().__init__(f"Unknown material '{query}' in region '{region}'.")
