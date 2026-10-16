"""Unit tests for accepted-material parsing and public responses."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.model import Response, parse_accepted_materials

PLASTIC: dict[str, str] = {"slug": "plastique", "column": "PLST", "label": "Plastique"}
PLUGS_AND_PLASTIC: list[dict[str, str]] = [
    {"slug": "bouchons-plastique", "column": "BCHP", "label": "Bouchons plastique"},
    PLASTIC,
]


def _dump(response: Response) -> list[dict[str, str]] | None:
    """Return the accepted materials of a response as plain dictionaries."""
    if response.accepted_materials is None:
        return None
    return [item.model_dump() for item in response.accepted_materials]


def test_parse_accepted_materials_preserves_one_record() -> None:
    """One accepted material is parsed with its exact keys."""
    parsed = parse_accepted_materials({"accepted_materials": [PLASTIC]})
    assert parsed is not None
    assert [dict(item) if isinstance(item, dict) else item.model_dump() for item in parsed] == [
        PLASTIC
    ]


def test_parse_accepted_materials_preserves_several_records() -> None:
    """Several accepted materials keep their order and values."""
    parsed = parse_accepted_materials({"accepted_materials": PLUGS_AND_PLASTIC})
    assert parsed is not None
    assert [dict(item) if isinstance(item, dict) else item.model_dump() for item in parsed] == (
        PLUGS_AND_PLASTIC
    )


def test_parse_accepted_materials_returns_none_for_null() -> None:
    """A null value stays null."""
    assert parse_accepted_materials({"accepted_materials": None}) is None


def test_parse_accepted_materials_returns_none_for_invalid_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid accepted-material records are normalized to null with a warning."""
    with caplog.at_level(logging.WARNING):
        assert parse_accepted_materials({"accepted_materials": [{"slug": "inconnu"}]}) is None
    assert caplog.records


def test_parse_accepted_materials_ignores_legacy_collection_fields() -> None:
    """Legacy collection fields are never used to build accepted materials."""
    payload: dict[str, Any] = {
        "collection_material_code": "PLST",
        "collection_material_name": "Plastique",
        "collection_materials": ["plastique"],
    }
    assert parse_accepted_materials(payload) is None


def test_response_exposes_accepted_materials() -> None:
    """A response exposes several exact accepted-material records."""
    response = Response(
        material_name="Bouchon plastique",
        bin_color="Jaune",
        accepted_materials=PLUGS_AND_PLASTIC,
    )
    assert response.material_name == "Bouchon plastique"
    assert response.bin_color == "Jaune"
    assert _dump(response) == PLUGS_AND_PLASTIC
    assert not any(key.startswith("collection_") for key in response.model_dump())


def test_response_allows_explicit_null() -> None:
    """An out-of-catalog response serializes accepted materials as null."""
    response = Response(material_name="Litière chat", bin_color="Noir", accepted_materials=None)
    assert _dump(response) is None
    assert response.model_dump()["accepted_materials"] is None
