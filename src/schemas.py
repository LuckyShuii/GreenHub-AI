"""Pydantic models for waste metadata and vector indexing."""

from __future__ import annotations

from typing import Annotated, Literal, Self

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


class CandidateResponse(BaseModel):
    """Response returned when classification is below the confidence threshold."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_choice"] = "needs_choice"
    requires_user_choice: bool = True
    candidates: list[MaterialCandidate]


class MatchResponse(BaseModel):
    """Response returned when classification is confident enough."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["matched"] = "matched"
    material_name: str
    bin_color: str
    accepted_materials: list[AcceptedMaterial] | None = None
    query: str | None = None
    match_type: Literal["exact", "partial", "fuzzy"] | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class NameCandidate(BaseModel):
    """A ranked manual name-search candidate."""

    model_config = ConfigDict(extra="forbid")

    nom: str
    poubelle: str
    accepted_materials: list[AcceptedMaterial] | None = None
    score: float = Field(ge=0.0, le=1.0)


class NameCandidateResponse(BaseModel):
    """Response returned when manual name search requires a choice."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_choice"] = "needs_choice"
    query: str
    candidates: list[NameCandidate]


ManualConfirmResponse = Annotated[MatchResponse | NameCandidateResponse, Field(discriminator="status")]


class ManualMaterialRequest(BaseModel):
    """Body of the manual confirm endpoint."""

    model_config = ConfigDict(extra="forbid")

    nom: str = Field(..., min_length=1, max_length=100)
    region: str = Field(..., min_length=1)
    slug: str | None = Field(default=None, min_length=1, deprecated=True)

    @field_validator("nom")
    @classmethod
    def strip_name(cls, value: str) -> str:
        """Strip surrounding whitespace and reject an empty name."""
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank.")
        return value

    @field_validator("region")
    @classmethod
    def strip_region(cls, value: str) -> str:
        """Strip surrounding whitespace and reject an empty region."""
        value = value.strip()
        if not value:
            raise ValueError("Region must not be blank.")
        return value

    @field_validator("slug")
    @classmethod
    def strip_slug(cls, value: str | None) -> str | None:
        """Strip surrounding whitespace from a deprecated catalog slug."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Slug must not be blank when provided.")
        return value


class UnknownMaterialError(Exception):
    """Raised when a manual lookup does not match any known material."""

    def __init__(self, query: str, region: str, suggestions: list[NameCandidate] | None = None) -> None:
        """Store the query, region and any ranked suggestions."""
        self.query = query
        self.region = region
        self.suggestions = suggestions or []
        super().__init__(f"Unknown material '{query}' in region '{region}'.")
