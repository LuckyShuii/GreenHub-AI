"""Region discovery and loading from the data directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from logging_config import get_logger

from .materials import _normalize
from .schemas import QdrantPayload, WasteItem

logger = get_logger(__name__)


def discover_region_files(data_dir: Path) -> list[Path]:
    """Discover all region JSON files in the data directory."""
    if not data_dir.exists():
        msg = f"Data directory not found: {data_dir}"
        raise FileNotFoundError(msg)
    files = sorted(data_dir.glob("*.json"))
    logger.info("Discovered %d region file(s) in %s.", len(files), data_dir)
    return files


def load_region_payloads(json_path: Path) -> list[QdrantPayload]:
    """Load and validate the waste payloads declared in a region file."""
    region_name = json_path.stem
    try:
        raw_content: Any = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"Invalid JSON in {json_path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(raw_content, list):
        msg = f"JSON root must be a list in {json_path}."
        raise TypeError(msg)

    payloads: list[QdrantPayload] = []
    for index, entry in enumerate(raw_content):
        try:
            item = WasteItem.model_validate(entry)
        except ValidationError as error:
            logger.warning(
                "Skipping invalid entry %d in %s: %s", index, json_path.name, error
            )
            continue
        payloads.append(
            QdrantPayload(
                nom=item.nom,
                region=region_name,
                poubelle=item.poubelle,
                accepted_materials=item.accepted_materials,
            )
        )
    logger.info("Loaded %d items for region '%s'.", len(payloads), region_name)
    return payloads


def lookup_by_name(region: str, name: str) -> dict[str, Any] | None:
    """Find a waste entry in a region file by name (case/accent insensitive).

    Returns the raw entry dict (suitable for building a Response) or None.
    """
    if not name:
        return None
    needle = _normalize(name)
    candidates = [path for path in discover_region_files(Path(".")) if path.stem == region]
    if not candidates:
        # Fallback: scan all region files for the requested region name.
        candidates = list(Path(".").glob(f"{region}.json"))
    for json_path in candidates:
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            entry_name = entry.get("nom")
            if isinstance(entry_name, str) and _normalize(entry_name) == needle:
                return entry
    return None
