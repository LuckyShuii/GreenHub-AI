"""Tests for exact, partial and fuzzy region-catalog search."""

import json
from pathlib import Path

import pytest

from src.regions import RegionCatalog, UnknownRegionError


@pytest.fixture
def catalog(tmp_path: Path) -> RegionCatalog:
    """Return a catalog with deliberate similar-name ambiguities."""
    (tmp_path / "ile_de_france.json").write_text(
        json.dumps(
            [
                {"nom": "Pot de yaourt", "poubelle": "Jaune"},
                {"nom": "Bouteille plastique", "poubelle": "Jaune"},
                {"nom": "Verre cassé", "poubelle": "Vert"},
                {"nom": "Verre à boire cassé", "poubelle": "Vert"},
                {"nom": "Bouteille en verre", "poubelle": "Vert"},
            ]
        ),
        encoding="utf-8",
    )
    return RegionCatalog.from_directory(tmp_path)


def test_exact_and_unaccented_case_variant_match(catalog: RegionCatalog) -> None:
    """Exact canonical names and accentless case variants are exact matches."""
    assert catalog.search("Pot de yaourt", "ile_de_france").match_type == "exact"
    assert catalog.search("verre casse", "ile_de_france").match_type == "exact"


def test_partial_search_finds_yaourt(catalog: RegionCatalog) -> None:
    """A short constituent word can resolve to the unique relevant entry."""
    assert catalog.search("yaourt", "ile_de_france").status == "matched"
    assert catalog.search("yaourt", "ile_de_france").match_type == "partial"


def test_fuzzy_typo_is_suggested(catalog: RegionCatalog) -> None:
    """A typo produces a ranked suggestion instead of an unrelated match."""
    result = catalog.search("boutielle plastique", "ile_de_france")
    assert result.status in {"matched", "needs_choice"}
    assert result.entry is not None or any(c.nom == "Bouteille plastique" for c in result.candidates)


def test_ambiguous_partial_requires_choice(catalog: RegionCatalog) -> None:
    """A shared name is not silently resolved when similar candidates exist."""
    result = catalog.search("verre", "ile_de_france")
    assert result.status == "needs_choice"
    assert {candidate.nom for candidate in result.candidates} >= {"Verre cassé", "Verre à boire cassé"}


def test_none_and_short_queries(catalog: RegionCatalog) -> None:
    """Unrelated and too-short input is not accepted as a match."""
    assert catalog.search("voiture", "ile_de_france").status == "not_found"
    assert catalog.search("ab", "ile_de_france").status == "not_found"


def test_thresholds_control_match_decision(catalog: RegionCatalog) -> None:
    """Custom thresholds can prevent automatic matching."""
    result = catalog.search("yaourt", "ile_de_france", accept_threshold=0.99)
    assert result.status == "needs_choice"


def test_candidate_order_is_stable(catalog: RegionCatalog) -> None:
    """Equivalent searches return candidates in deterministic order."""
    first = catalog.search("verre casse", "ile_de_france")
    second = catalog.search("verre casse", "ile_de_france")
    assert first.candidates == second.candidates


def test_unknown_region_is_distinct(catalog: RegionCatalog) -> None:
    """A missing region raises a dedicated error."""
    with pytest.raises(UnknownRegionError):
        catalog.search("verre", "absent")


def test_every_real_catalog_name_matches_exact() -> None:
    """Every shipped regional label is found as an exact match."""
    root = Path(__file__).parent
    catalog = RegionCatalog.from_directory(root)
    source = json.loads((root / "ile_de_france.json").read_text(encoding="utf-8"))
    for item in source:
        assert catalog.search(item["nom"], "ile_de_france").match_type == "exact"
