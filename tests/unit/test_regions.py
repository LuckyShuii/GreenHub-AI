"""Unit tests for region discovery and payload loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.regions import discover_region_files, load_region_payloads


class TestDiscoverRegionFiles:
    """Behaviour of the region file discovery helper."""

    def test_returns_sorted_json_files(self, tmp_path: Path) -> None:
        """Only JSON files are returned, in lexicographic order.

        Args:
            tmp_path: Temporary directory fixture.
        """
        for name in ["occitanie.json", "bretagne.json", "notes.txt"]:
            (tmp_path / name).write_text("[]", encoding="utf-8")

        files = discover_region_files(tmp_path)

        assert [path.name for path in files] == [
            "bretagne.json",
            "occitanie.json",
        ]

    def test_returns_empty_list_for_empty_directory(
        self, tmp_path: Path
    ) -> None:
        """An existing but empty directory yields no files.

        Args:
            tmp_path: Temporary directory fixture.
        """
        assert discover_region_files(tmp_path) == []

    def test_raises_when_directory_missing(self, tmp_path: Path) -> None:
        """A missing data directory raises FileNotFoundError.

        Args:
            tmp_path: Temporary directory fixture.
        """
        with pytest.raises(FileNotFoundError):
            discover_region_files(tmp_path / "absent")


class TestLoadRegionPayloads:
    """Behaviour of the region payload loader."""

    def test_loads_all_valid_entries(self, region_file: Path) -> None:
        """Every valid entry is converted into a payload.

        Args:
            region_file: Valid region JSON fixture.
        """
        payloads = load_region_payloads(region_file)

        assert len(payloads) == 3
        assert {payload.nom for payload in payloads} == {
            "bouteille en verre",
            "carton",
            "reste alimentaire",
        }

    def test_derives_region_from_filename(self, region_file: Path) -> None:
        """The region name comes from the file stem, not the content.

        Args:
            region_file: Valid region JSON fixture.
        """
        payloads = load_region_payloads(region_file)
        assert all(
            payload.region == "ile-de-france" for payload in payloads
        )

    def test_skips_invalid_entries(
        self, malformed_region_file: Path
    ) -> None:
        """Invalid entries are skipped without aborting the load.

        Args:
            malformed_region_file: Partially invalid region fixture.
        """
        payloads = load_region_payloads(malformed_region_file)

        assert len(payloads) == 1
        assert payloads[0].nom == "carton"

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Syntactically broken JSON raises a ValueError.

        Args:
            tmp_path: Temporary directory fixture.
        """
        path = tmp_path / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_region_payloads(path)

    def test_raises_when_root_is_not_a_list(self, tmp_path: Path) -> None:
        """A non-list JSON root raises a ValueError.

        Args:
            tmp_path: Temporary directory fixture.
        """
        path = tmp_path / "object.json"
        path.write_text(json.dumps({"nom": "carton"}), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a list"):
            load_region_payloads(path)

    def test_returns_empty_list_for_empty_array(
        self, tmp_path: Path
    ) -> None:
        """An empty JSON array produces no payloads.

        Args:
            tmp_path: Temporary directory fixture.
        """
        path = tmp_path / "vide.json"
        path.write_text("[]", encoding="utf-8")
        assert load_region_payloads(path) == []
