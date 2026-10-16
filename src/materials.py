"""Helpers to build accepted materials from the catalog."""

from __future__ import annotations

from .schemas import MATERIAL_CATALOG, AcceptedMaterial, MaterialSlug


def build_material(slug: MaterialSlug) -> AcceptedMaterial:
    """Return the catalog material matching a slug."""
    column, label = MATERIAL_CATALOG[slug]
    return AcceptedMaterial(slug=slug, column=column, label=label)


def build_materials(slugs: list[MaterialSlug]) -> list[AcceptedMaterial] | None:
    """Return the catalog materials for slugs, or None when empty."""
    if not slugs:
        return None
    return [build_material(slug) for slug in slugs]
