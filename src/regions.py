"""Discover region data and provide a precomputed searchable name catalog."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from logging_config import get_logger

from .schemas import NameCandidate, QdrantPayload, WasteItem
from .text_matching import normalize, score, tokenize

_REGION_SEPARATORS: re.Pattern[str] = re.compile(r"[\s_\-']+")
logger = get_logger(__name__)


@dataclass(frozen=True)
class _CatalogEntry:
    """Validated data and precomputed search forms for one waste item."""

    name: str
    bin_color: str
    accepted_materials: tuple[Any, ...] | None
    normalized_name: str
    tokens: frozenset[str]

    def candidate(self, value: float) -> NameCandidate:
        """Build a public choice candidate with its search score."""
        return NameCandidate(
            nom=self.name,
            poubelle=self.bin_color,
            accepted_materials=list(self.accepted_materials) if self.accepted_materials else None,
            score=round(value, 4),
        )


@dataclass(frozen=True)
class SearchResult:
    """Represent a catalog search outcome and its ranked candidates."""

    status: str
    query: str
    match_type: str | None = None
    score: float | None = None
    entry: _CatalogEntry | None = None
    candidates: tuple[NameCandidate, ...] = ()


class UnknownRegionError(Exception):
    """Raised when a region identifier is absent from the loaded catalogs."""

    def __init__(self, region: str) -> None:
        """Store the missing region identifier."""
        self.region = region
        super().__init__(f"Unknown region '{region}'.")


def normalize_region_key(value: str) -> str:
    """Return a canonical key for comparing region identifiers."""
    decomposed: str = unicodedata.normalize("NFKD", value)
    ascii_only: str = decomposed.encode("ascii", "ignore").decode("ascii")
    folded: str = ascii_only.casefold().strip()
    return _REGION_SEPARATORS.sub("_", folded).strip("_")


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
            logger.warning("Skipping invalid entry %d in %s: %s", index, json_path.name, error)
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


class RegionCatalog:
    """Frozen, validated set of region entries with precomputed search data."""

    def __init__(self, regions: dict[str, tuple[_CatalogEntry, ...]]) -> None:
        """Create an immutable catalog mapping normalized region IDs to entries."""
        self._regions = MappingProxyType(dict(regions))

    @classmethod
    def from_directory(cls, data_dir: Path) -> RegionCatalog:
        """Load and validate all JSON region catalogs from a directory."""
        regions: dict[str, tuple[_CatalogEntry, ...]] = {}
        for path in discover_region_files(data_dir):
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}: {error}") from error
            if not isinstance(raw, list):
                raise TypeError(f"JSON root must be a list in {path}.")
            entries: list[_CatalogEntry] = []
            for index, item_raw in enumerate(raw):
                try:
                    item = WasteItem.model_validate(item_raw)
                except ValidationError as error:
                    logger.warning("Skipping invalid entry %d in %s: %s", index, path.name, error)
                    continue
                entries.append(
                    _CatalogEntry(
                        name=item.nom,
                        bin_color=item.poubelle,
                        accepted_materials=tuple(item.accepted_materials) if item.accepted_materials else None,
                        normalized_name=normalize(item.nom),
                        tokens=frozenset(tokenize(item.nom)),
                    )
                )
            regions[normalize_region_key(path.stem)] = tuple(entries)
        return cls(regions)

    @property
    def regions(self) -> tuple[str, ...]:
        """Return the loaded region identifiers in stable sorted order."""
        return tuple(sorted(self._regions))

    def search(
        self,
        query: str,
        region: str,
        *,
        accept_threshold: float = 0.85,
        candidate_threshold: float = 0.60,
        ambiguity_margin: float = 0.05,
        max_candidates: int = 5,
        min_query_length: int = 3,
    ) -> SearchResult:
        """Search a region using exact, partial, then fuzzy matching."""
        region_key = normalize_region_key(region)
        if region_key not in self._regions:
            raise UnknownRegionError(region)
        query_tokens = tokenize(query)
        normalized_query = normalize(query)
        if len(normalized_query) < min_query_length or not query_tokens:
            return SearchResult(status="not_found", query=query)
        entries = self._regions[region_key]
        exact = [entry for entry in entries if entry.normalized_name == normalized_query]
        if len(exact) == 1:
            return SearchResult(status="matched", query=query, match_type="exact", score=1.0, entry=exact[0])
        if len(exact) > 1:
            candidates = tuple(entry.candidate(1.0) for entry in exact[:max_candidates])
            return SearchResult(status="needs_choice", query=query, candidates=candidates)

        query_token_set = set(query_tokens)
        partials: list[tuple[float, _CatalogEntry]] = []
        for entry in entries:
            if normalized_query in entry.normalized_name or entry.tokens.issubset(query_token_set):
                coverage = len(entry.tokens & query_token_set) / max(1, len(entry.tokens))
                query_coverage = len(entry.tokens & query_token_set) / max(1, len(query_token_set))
                value = min(0.95, 0.85 + 0.10 * max(coverage, query_coverage))
                partials.append((value, entry))
        if partials:
            ranked = sorted(partials, key=lambda item: (-item[0], item[1].name.casefold()))
            return self._decide(
                query, ranked, "partial", accept_threshold, candidate_threshold, ambiguity_margin, max_candidates
            )

        ranked = sorted(
            ((score(query, entry.name), entry) for entry in entries),
            key=lambda item: (-item[0], item[1].name.casefold()),
        )
        return self._decide(
            query, ranked, "fuzzy", accept_threshold, candidate_threshold, ambiguity_margin, max_candidates
        )

    @staticmethod
    def _decide(
        query: str,
        ranked: list[tuple[float, _CatalogEntry]],
        match_type: str,
        accept_threshold: float,
        candidate_threshold: float,
        ambiguity_margin: float,
        max_candidates: int,
    ) -> SearchResult:
        """Choose an unambiguous match or return stable ranked suggestions."""
        if not ranked:
            return SearchResult(status="not_found", query=query)
        top_score, top_entry = ranked[0]
        gap = top_score - ranked[1][0] if len(ranked) > 1 else 1.0
        if top_score >= accept_threshold and gap >= ambiguity_margin:
            return SearchResult(status="matched", query=query, match_type=match_type, score=top_score, entry=top_entry)
        qualified = [(value, entry) for value, entry in ranked if value >= candidate_threshold]
        candidate_pool = qualified if qualified else ranked[:3]
        candidates = tuple(entry.candidate(value) for value, entry in candidate_pool[:max_candidates])
        return SearchResult(status="needs_choice" if qualified else "not_found", query=query, candidates=candidates)


def lookup_by_name(region: str, name: str) -> dict[str, Any] | None:
    """Find an exact normalized name in a single local region JSON file."""
    if not name:
        return None
    path = Path(f"{region}.json")
    if not path.is_file():
        return None
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    if not isinstance(raw, list):
        return None
    needle = normalize(name)
    return next(
        (
            entry
            for entry in raw
            if isinstance(entry, dict) and isinstance(entry.get("nom"), str) and normalize(entry["nom"]) == needle
        ),
        None,
    )
