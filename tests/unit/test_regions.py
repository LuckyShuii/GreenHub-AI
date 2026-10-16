"""Unit tests for region discovery and payload loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.regions import discover_region_files, load_region_payloads

PLASTIC: dict[str, str] = {"slug": "plastique", "column": "PLST", "label": "Plastique"}


def _write_region(path: Path, entries: list[dict[str, object]]) -> None:
    """Write region entries to a UTF-8 JSON fixture."""
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


class TestDiscoverRegionFiles:
    """Behaviour of the region file discovery helper."""

    def test_returns_sorted_json_files(self, tmp_path: Path) -> None:
        """Only JSON files are returned, in lexicographic order."""
        for name in ("occitanie.json", "bretagne.json", "notes.txt"):
            (tmp_path / name).write_text("[]", encoding="utf-8")
        files = discover_region_files(tmp_path)
        assert [path.name for path in files] == ["bretagne.json", "occitanie.json"]

    def test_returns_empty_list_for_empty_directory(self, tmp_path: Path) -> None:
        """An existing but empty directory yields no files."""
        assert discover_region_files(tmp_path) == []

    def test_raises_when_directory_missing(self, tmp_path: Path) -> None:
        """A missing data directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_region_files(tmp_path / "absent")


class TestLoadRegionPayloads:
    """Behaviour of the region payload loader."""

    def test_loads_one_accepted_material(self, tmp_path: Path) -> None:
        """A catalog match yields one exact accepted-material record."""
        path = tmp_path / "ile_de_france.json"
        _write_region(
            path,
            [
                {
                    "nom": "Film étirable alimentaire",
                    "poubelle": "Noir",
                    "accepted_materials": [PLASTIC],
                }
            ],
        )
        payloads = load_region_payloads(path)
        assert len(payloads) == 1
        assert payloads[0].nom == "Film étirable alimentaire"
        assert payloads[0].poubelle == "Noir"
        assert payloads[0].accepted_materials is not None
        assert [m.model_dump() for m in payloads[0].accepted_materials] == [PLASTIC]

    def test_loads_several_accepted_materials(self, tmp_path: Path) -> None:
        """A compound match preserves every material in order."""
        path = tmp_path / "ile_de_france.json"
        materials = [
            {"slug": "bouchons-plastique", "column": "BCHP", "label": "Bouchons plastique"},
            PLASTIC,
        ]
        _write_region(
            path,
            [{"nom": "Bouchon plastique", "poubelle": "Jaune", "accepted_materials": materials}],
        )
        payloads = load_region_payloads(path)
        assert len(payloads) == 1
        assert payloads[0].accepted_materials is not None
        assert [m.model_dump() for m in payloads[0].accepted_materials] == materials

    def test_loads_explicit_null_for_out_of_catalog_item(self, tmp_path: Path) -> None:
        """An out-of-catalog item keeps a null accepted-material value."""
        path = tmp_path / "ile_de_france.json"
        _write_region(
            path,
            [{"nom": "Litière chat", "poubelle": "Noir", "accepted_materials": None}],
        )
        payloads = load_region_payloads(path)
        assert len(payloads) == 1
        assert payloads[0].nom == "Litière chat"
        assert payloads[0].poubelle == "Noir"
        assert payloads[0].accepted_materials is None

    def test_skips_entries_with_legacy_collection_fields(self, tmp_path: Path) -> None:
        """Entries still using collection_* fields are skipped."""
        path = tmp_path / "ile_de_france.json"
        _write_region(
            path,
            [
                {
                    "nom": "Film étirable alimentaire",
                    "poubelle": "Noir",
                    "collection_material_code": "PLST",
                    "collection_material_name": "Plastique",
                    "collection_materials": ["Plastique"],
                }
            ],
        )
        assert load_region_payloads(path) == []

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Syntactically broken JSON raises a ValueError."""
        path = tmp_path / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_region_payloads(path)

    def test_raises_when_root_is_not_a_list(self, tmp_path: Path) -> None:
        """A non-list JSON root raises a TypeError."""
        path = tmp_path / "object.json"
        path.write_text(json.dumps({"nom": "carton"}), encoding="utf-8")
        with pytest.raises(TypeError, match="must be a list"):
            load_region_payloads(path)

    def test_returns_empty_list_for_empty_array(self, tmp_path: Path) -> None:
        """An empty JSON array produces no payloads."""
        path = tmp_path / "vide.json"
        path.write_text("[]", encoding="utf-8")
        assert load_region_payloads(path) == []
