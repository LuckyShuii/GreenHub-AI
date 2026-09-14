"""Region discovery and loading from the data directory."""

import json
from pathlib import Path

from pydantic import ValidationError

from logging_config import get_logger
from .schemas import QdrantPayload

logger = get_logger(__name__)


def discover_region_files(data_dir: Path) -> list[Path]:
    """Discover all region JSON files in the data directory.

    Args:
        data_dir: Directory expected to contain region JSON files.

    Returns:
        A sorted list of JSON file paths.

    Raises:
        FileNotFoundError: If the data directory does not exist.

    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.json"))
    logger.info("Discovered %d region file(s) in %s.", len(files), data_dir)
    return files


def load_region_payloads(json_path: Path) -> list[QdrantPayload]:
    """Load and validate waste payloads from a region JSON file.

    Args:
        json_path: Path to the region JSON file.

    Returns:
        A list of validated QdrantPayload instances.

    Raises:
        ValueError: If the JSON content is malformed.

    """
    region_name = json_path.stem
    try:
        raw_content = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {json_path}: {error}") from error

    if not isinstance(raw_content, list):
        raise ValueError(f"JSON root must be a list in {json_path}.")

    payloads: list[QdrantPayload] = []
    for index, entry in enumerate(raw_content):
        try:
            payload = QdrantPayload(
                nom=entry["nom"],
                region=region_name,
                poubelle=entry["poubelle"],
            )
            payloads.append(payload)
        except (KeyError, ValidationError) as error:
            logger.warning(
                "Skipping invalid entry %d in %s: %s",
                index,
                json_path.name,
                error,
            )

    logger.info("Loaded %d items for region '%s'.", len(payloads), region_name)
    return payloads
