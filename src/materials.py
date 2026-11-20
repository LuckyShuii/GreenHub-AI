"""Helpers to build accepted materials from the catalog."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .schemas import (
    MATERIAL_CATALOG,
    MATERIAL_LABELS,
    AcceptedMaterial,
    MaterialCandidate,
    MaterialSlug,
)


def build_material(slug: MaterialSlug) -> AcceptedMaterial:
    """Return the catalog material matching a slug."""
    column, label = MATERIAL_CATALOG[slug]
    return AcceptedMaterial(slug=slug, column=column, label=label)


def build_materials(slugs: list[MaterialSlug]) -> list[AcceptedMaterial] | None:
    """Return the catalog materials for slugs, or None when empty."""
    if not slugs:
        return None
    return [build_material(slug) for slug in slugs]


def _normalize(value: str) -> str:
    """Lowercase and strip simple French accents for fuzzy name lookup."""
    lowered = value.strip().lower()
    accents = str.maketrans(
        {
            "à": "a",
            "â": "a",
            "ä": "a",
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ö": "o",
            "ù": "u",
            "û": "u",
            "ü": "u",
            "ç": "c",
            "ÿ": "y",
        }
    )
    return lowered.translate(accents)


def lookup_by_name(name: str) -> MaterialSlug | None:
    """Resolve a slug from a free-text material name or slug.

    Matches first against the slug itself, then against the display label.
    Comparison is case-insensitive and accent-insensitive.
    """
    if not name:
        return None
    needle = _normalize(name)
    if needle in MATERIAL_CATALOG:
        return needle  # type: ignore[return-value]
    for slug, label in MATERIAL_LABELS.items():
        if _normalize(label) == needle:
            return slug  # type: ignore[return-value]
    return None


def candidate_from_payload(payload: dict[str, Any]) -> MaterialCandidate:
    """Build a candidate object from a Qdrant payload.

    The internal similarity score is never read here; the score is filtered
    out so it never reaches the response.
    """
    slug_value = None
    label_value = None
    accepted_raw = payload.get("accepted_materials")
    if isinstance(accepted_raw, list) and accepted_raw:
        first = accepted_raw[0]
        if isinstance(first, dict):
            slug_value = first.get("slug")
            label_value = first.get("label")
    accepted = None
    try:
        accepted = MaterialCandidate.model_validate(
            {"accepted_materials": accepted_raw}
        ).accepted_materials
    except ValidationError:
        accepted = None
    return MaterialCandidate(
        nom=str(payload.get("nom", "unknown")),
        poubelle=str(payload.get("poubelle", "unknown")),
        slug=slug_value,
        label=label_value,
        accepted_materials=accepted,
    )


def deduplicate_candidates(
    candidates: list[MaterialCandidate],
) -> list[MaterialCandidate]:
    """Return candidates deduplicated by material name, preserving order."""
    seen: set[str] = set()
    deduped: list[MaterialCandidate] = []
    for candidate in candidates:
        key = candidate.nom.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
