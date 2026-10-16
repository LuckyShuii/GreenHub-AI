"""Unit tests for the material catalog and waste schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    MATERIAL_CATALOG,
    AcceptedMaterial,
    MaterialMetadata,
    WasteItem,
)

EXPECTED_CATALOG: dict[str, tuple[str, str]] = {
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
LEGACY_FIELDS: tuple[str, ...] = (
    "collection_material_code",
    "collection_material_name",
    "collection_materials",
)


class TestMaterialCatalog:
    """Validation of the public material catalog."""

    def test_contains_the_complete_catalog(self) -> None:
        """The catalog exposes all supported slugs, columns and labels."""
        assert MATERIAL_CATALOG == EXPECTED_CATALOG


class TestAcceptedMaterial:
    """Validation behaviour of accepted material records."""

    @pytest.mark.parametrize(("slug", "values"), EXPECTED_CATALOG.items())
    def test_accepts_catalog_record(self, slug: str, values: tuple[str, str]) -> None:
        """A catalog slug with its matching column and label is accepted."""
        column, label = values
        material = AcceptedMaterial(slug=slug, column=column, label=label)
        assert material.model_dump() == {"slug": slug, "column": column, "label": label}

    def test_rejects_inconsistent_catalog_record(self) -> None:
        """A known slug cannot be paired with another material's metadata."""
        with pytest.raises(ValidationError):
            AcceptedMaterial(slug="plastique", column="VER", label="Verre")

    def test_rejects_unknown_slug(self) -> None:
        """Slugs outside the catalog are rejected."""
        with pytest.raises(ValidationError):
            AcceptedMaterial(slug="litiere-chat", column="CAT", label="Litière")

    def test_rejects_extra_fields(self) -> None:
        """Material records accept exactly the slug, column and label keys."""
        with pytest.raises(ValidationError):
            AcceptedMaterial(slug="verre", column="VER", label="Verre", code="VER")


class TestMaterialMetadata:
    """Validation behaviour of material metadata."""

    def test_normalizes_empty_materials_to_none(self) -> None:
        """An empty accepted-material list is represented as null."""
        assert MaterialMetadata(accepted_materials=[]).accepted_materials is None

    def test_preserves_one_accepted_material(self) -> None:
        """A single accepted material is preserved with its exact keys."""
        material = {"slug": "plastique", "column": "PLST", "label": "Plastique"}
        metadata = MaterialMetadata(accepted_materials=[material])
        assert metadata.accepted_materials is not None
        assert [item.model_dump() for item in metadata.accepted_materials] == [material]

    def test_preserves_several_accepted_materials(self) -> None:
        """Multiple accepted materials keep their order and values."""
        materials = [
            {"slug": "bouchons-plastique", "column": "BCHP", "label": "Bouchons plastique"},
            {"slug": "plastique", "column": "PLST", "label": "Plastique"},
        ]
        metadata = MaterialMetadata(accepted_materials=materials)
        assert metadata.accepted_materials is not None
        assert [item.model_dump() for item in metadata.accepted_materials] == materials


class TestWasteItem:
    """Validation behaviour of the public waste item model."""

    def test_preserves_nom_and_poubelle(self) -> None:
        """Waste names and bin colors remain unchanged."""
        item = WasteItem(
            nom="Film étirable alimentaire",
            poubelle="Noir",
            accepted_materials=[
                {"slug": "plastique", "column": "PLST", "label": "Plastique"}
            ],
        )
        assert item.nom == "Film étirable alimentaire"
        assert item.poubelle == "Noir"
        assert set(item.model_dump()) == {"nom", "poubelle", "accepted_materials"}

    def test_allows_explicit_null_for_out_of_catalog_waste(self) -> None:
        """Out-of-catalog waste carries a null accepted-material value."""
        item = WasteItem(nom="Litière chat", poubelle="Noir", accepted_materials=None)
        assert item.accepted_materials is None
        assert item.model_dump()["accepted_materials"] is None

    @pytest.mark.parametrize("field", LEGACY_FIELDS)
    def test_rejects_legacy_collection_fields(self, field: str) -> None:
        """Each legacy collection field is rejected by the schema."""
        with pytest.raises(ValidationError):
            WasteItem(nom="Litière chat", poubelle="Noir", **{field: "Plastique"})
