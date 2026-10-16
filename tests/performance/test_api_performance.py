"""Performance and schema tests for the production waste classification endpoint."""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from tqdm import tqdm

DEFAULT_URL: str = "https://greenhub.lucasboillot.fr/ai/greener/upload/dechets"
TARGET_URL: str = os.environ.get("GREENER_API_URL", DEFAULT_URL)
UPLOAD_COUNT: int = 10
REQUESTS_PER_UPLOAD: int = 150
MIN_REQUESTS_PER_SECOND: float = 50.0
CONCURRENCY: int = 16
TARGET_REGION: str = "ile_de_france"
DATA_DIR: Path = Path(__file__).parents[1] / "data"
OIL_IMAGE_PATH: Path = DATA_DIR / "test.jpeg"
BATTERY_IMAGE_PATH: Path = DATA_DIR / "test2.jpg"
BENCHMARK_IMAGE_PATH: Path = OIL_IMAGE_PATH
ACCEPTED_STATUS: frozenset[int] = frozenset({200, 404, 422})
CONNECT_TIMEOUT: float = 10.0
READ_TIMEOUT: float = 30.0
CONNECT_RETRIES: int = 2
MAX_REPORTED_ERRORS: int = 5

MATERIAL_CATALOG: dict[str, tuple[str, str]] = {
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
MATERIAL_KEYS: frozenset[str] = frozenset({"slug", "column", "label"})
REQUIRED_TEXT_FIELDS: tuple[str, ...] = ("material_name", "bin_color")
LEGACY_FIELDS: tuple[str, ...] = (
    "collection_material_code",
    "collection_material_name",
    "collection_materials",
)
IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


@dataclass(frozen=True)
class ImageCase:
    """Expected production result for one uploaded test image."""

    path: Path
    name_keywords: tuple[str, ...]
    expected_slugs: frozenset[str] | None


IMAGE_CASES: tuple[ImageCase, ...] = (
    ImageCase(
        path=OIL_IMAGE_PATH,
        name_keywords=("huile",),
        expected_slugs=None,
    ),
    ImageCase(
        path=BATTERY_IMAGE_PATH,
        name_keywords=("batterie", "pile", "lithium"),
        expected_slugs=frozenset({"piles"}),
    ),
)


@dataclass
class BenchmarkReport:
    """Aggregated timing, transport and schema results of a benchmark run."""

    latencies: list[float] = field(default_factory=list)
    failures: int = 0
    timeouts: int = 0
    schema_failures: int = 0
    list_results: int = 0
    null_results: int = 0
    errors: set[str] = field(default_factory=set)


def validate_material(material: Any) -> list[str]:
    """Validate a single accepted material object against the catalog.

    Args:
        material: Candidate material object.

    Returns:
        A list of error messages, empty when the material is valid.
    """
    if not isinstance(material, Mapping):
        return [f"material is not an object: {material!r}"]
    keys = set(material)
    if keys != MATERIAL_KEYS:
        return [f"material keys {sorted(keys)} differ from {sorted(MATERIAL_KEYS)}"]
    slug = material["slug"]
    if slug not in MATERIAL_CATALOG:
        return [f"unknown material slug {slug!r}"]
    column, label = MATERIAL_CATALOG[slug]
    errors: list[str] = []
    if material["column"] != column:
        errors.append(f"column {material['column']!r} does not match {column!r} for {slug!r}")
    if material["label"] != label:
        errors.append(f"label {material['label']!r} does not match {label!r} for {slug!r}")
    return errors


def validate_response_body(body: Any) -> list[str]:
    """Validate a classification response body against the new schema.

    Args:
        body: Decoded JSON response body.

    Returns:
        A list of error messages, empty when the body is valid.
    """
    if not isinstance(body, Mapping):
        return [f"body is not an object: {type(body).__name__}"]
    errors: list[str] = [
        f"field {name!r} must be a non-empty string"
        for name in REQUIRED_TEXT_FIELDS
        if not isinstance(body.get(name), str) or not body[name].strip()
    ]
    errors.extend(
        f"legacy field {name!r} is still present" for name in LEGACY_FIELDS if name in body
    )
    if "accepted_materials" not in body:
        errors.append("field 'accepted_materials' is missing")
        return errors
    materials = body["accepted_materials"]
    if materials is None:
        return errors
    if not isinstance(materials, list):
        errors.append("field 'accepted_materials' must be a list or null")
    elif not materials:
        errors.append("field 'accepted_materials' must be null instead of an empty list")
    else:
        for material in materials:
            errors.extend(validate_material(material))
    return errors


def load_image(path: Path) -> bytes:
    """Read a test image used for uploads, skipping the test when it is missing.

    Args:
        path: Location of the image file.

    Returns:
        Raw bytes of the image.
    """
    if not path.is_file():
        pytest.skip(f"test image not found at {path}")
    return path.read_bytes()


def build_files(path: Path, payload: bytes) -> dict[str, tuple[str, bytes, str]]:
    """Build the multipart payload for an image upload.

    Args:
        path: Location of the image file, used for its name and MIME type.
        payload: Raw bytes of the image.

    Returns:
        The multipart files mapping expected by httpx.
    """
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return {"file": (path.name, payload, mime_type)}


@pytest.fixture
async def production_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide an HTTP client configured for the remote production server.

    Yields:
        An asynchronous HTTP client with timeouts, retries and connection limits.
    """
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    transport = httpx.AsyncHTTPTransport(retries=CONNECT_RETRIES)
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        transport=transport,
        follow_redirects=True,
    ) as client:
        yield client


async def upload_image(client: httpx.AsyncClient, path: Path) -> httpx.Response:
    """Upload one image to the production endpoint.

    Args:
        client: HTTP client bound to the production server.
        path: Location of the image file to upload.

    Returns:
        The response returned by the endpoint.
    """
    files = build_files(path, load_image(path))
    try:
        return await client.post(TARGET_URL, files=files, params={"region": TARGET_REGION})
    except httpx.TimeoutException as exc:
        pytest.fail(f"upload of {path.name} to {TARGET_URL} timed out: {exc!r}")
    except httpx.HTTPError as exc:
        pytest.fail(f"network error while uploading {path.name} to {TARGET_URL}: {exc!r}")


async def send_request(
    client: httpx.AsyncClient,
    files: dict[str, tuple[str, bytes, str]],
    semaphore: asyncio.Semaphore,
    report: BenchmarkReport,
    progress: tqdm,
) -> None:
    """Send one benchmark request and record its latency and validity.

    Args:
        client: HTTP client bound to the production server.
        files: Multipart payload of the benchmark image.
        semaphore: Semaphore limiting concurrent requests.
        report: Report collecting the benchmark results.
        progress: Progress bar updated after each request.
    """
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(
                TARGET_URL, files=files, params={"region": TARGET_REGION}
            )
        except httpx.TimeoutException:
            report.timeouts += 1
            report.failures += 1
        except httpx.HTTPError as exc:
            report.failures += 1
            report.errors.add(f"network error: {type(exc).__name__}")
        else:
            record_response(response, report)
        finally:
            report.latencies.append(time.perf_counter() - start)
            progress.update(1)


def record_response(response: httpx.Response, report: BenchmarkReport) -> None:
    """Record the status and schema validity of a benchmark response.

    Args:
        response: Response returned by the endpoint.
        report: Report collecting the benchmark results.
    """
    if response.status_code not in ACCEPTED_STATUS:
        report.failures += 1
        report.errors.add(f"unexpected status {response.status_code}")
        return
    if response.status_code != httpx.codes.OK:
        return
    try:
        body: Any = response.json()
    except ValueError:
        report.schema_failures += 1
        report.errors.add("response body is not valid JSON")
        return
    errors = validate_response_body(body)
    if errors:
        report.schema_failures += 1
        report.errors.update(errors)
    elif body["accepted_materials"] is None:
        report.null_results += 1
    else:
        report.list_results += 1


async def run_benchmark(
    client: httpx.AsyncClient, path: Path
) -> tuple[BenchmarkReport, float]:
    """Run all benchmark requests against the production endpoint.

    Args:
        client: HTTP client bound to the production server.
        path: Location of the image uploaded by every request.

    Returns:
        The benchmark report and the total elapsed time in seconds.
    """
    files = build_files(path, load_image(path))
    report = BenchmarkReport()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = UPLOAD_COUNT * REQUESTS_PER_UPLOAD
    start = time.perf_counter()
    with tqdm(total=total, desc="requests") as progress:
        await asyncio.gather(
            *(send_request(client, files, semaphore, report, progress) for _ in range(total))
        )
    return report, time.perf_counter() - start


@pytest.mark.performance
async def test_api_throughput(production_client: httpx.AsyncClient) -> None:
    """The production endpoint sustains the target throughput with valid responses.

    Args:
        production_client: HTTP client bound to the production server.
    """
    report, elapsed = await run_benchmark(production_client, BENCHMARK_IMAGE_PATH)
    total = UPLOAD_COUNT * REQUESTS_PER_UPLOAD
    throughput = total / elapsed
    mean_latency = statistics.mean(report.latencies)
    p95_latency = statistics.quantiles(report.latencies, n=20)[-1]
    print(
        f"\n{total} requests in {elapsed:.2f}s ({throughput:.1f} req/s), "
        f"mean {mean_latency * 1000:.1f}ms, p95 {p95_latency * 1000:.1f}ms, "
        f"failures {report.failures}, timeouts {report.timeouts}, "
        f"schema failures {report.schema_failures}, "
        f"list {report.list_results}, null {report.null_results}"
    )
    reported = sorted(report.errors)[:MAX_REPORTED_ERRORS]
    assert report.failures == 0, f"{report.failures} failed requests: {reported}"
    assert report.schema_failures == 0, f"{report.schema_failures} invalid bodies: {reported}"
    assert throughput >= MIN_REQUESTS_PER_SECOND, (
        f"throughput {throughput:.1f} req/s below {MIN_REQUESTS_PER_SECOND} req/s"
    )


@pytest.mark.performance
@pytest.mark.parametrize("case", IMAGE_CASES, ids=lambda case: case.path.name)
async def test_image_upload_returns_expected_format(
    production_client: httpx.AsyncClient, case: ImageCase
) -> None:
    """Each test image yields a valid body with the expected accepted materials.

    Args:
        production_client: HTTP client bound to the production server.
        case: Image to upload and its expected result.
    """
    response = await upload_image(production_client, case.path)
    assert response.status_code == httpx.codes.OK, (
        f"unexpected status {response.status_code}: {response.text[:200]}"
    )
    assert response.headers.get("content-type", "").startswith("application/json")
    body: Any = response.json()
    errors = validate_response_body(body)
    assert errors == [], errors
    assert all(name not in body for name in LEGACY_FIELDS)
    name = body["material_name"].lower()
    assert any(keyword in name for keyword in case.name_keywords), (
        f"expected one of {case.name_keywords} in prediction, got {body['material_name']!r}"
    )
    materials = body["accepted_materials"]
    if case.expected_slugs is None:
        assert materials is None, f"expected null accepted_materials, got {materials!r}"
        return
    assert isinstance(materials, list) and materials, "expected a non-empty materials list"
    slugs = {material["slug"] for material in materials}
    assert case.expected_slugs <= slugs, (
        f"expected slugs {sorted(case.expected_slugs)} in {sorted(slugs)}"
    )


@pytest.mark.parametrize(
    "body",
    [
        {
            "material_name": "Bouteille",
            "bin_color": "Jaune",
            "accepted_materials": [
                {"slug": "plastique", "column": "PLST", "label": "Plastique"},
            ],
        },
        {
            "material_name": "Batterie lithium",
            "bin_color": "Noir",
            "accepted_materials": [
                {"slug": "piles", "column": "PIL", "label": "Piles"},
                {"slug": "telephones", "column": "TLF", "label": "Téléphones"},
            ],
        },
        {
            "material_name": "Huile de friture usagée",
            "bin_color": "Noir",
            "accepted_materials": None,
        },
    ],
)
def test_validate_response_body_accepts_valid_bodies(body: dict[str, Any]) -> None:
    """Valid bodies with one, several or null accepted materials pass validation.

    Args:
        body: Candidate response body.
    """
    assert validate_response_body(body) == []


@pytest.mark.parametrize(
    "body",
    [
        {"material_name": "Film", "bin_color": "Noir"},
        {"material_name": "Film", "bin_color": "Noir", "accepted_materials": []},
        {
            "material_name": "Film",
            "bin_color": "Noir",
            "accepted_materials": None,
            "collection_material_code": "PLST",
        },
        {
            "material_name": "Film",
            "bin_color": "Noir",
            "accepted_materials": [{"slug": "plastique", "column": "PLST"}],
        },
        {
            "material_name": "Film",
            "bin_color": "Noir",
            "accepted_materials": [
                {"slug": "plastique", "column": "PLST", "label": "Plastique", "extra": 1},
            ],
        },
        {
            "material_name": "Film",
            "bin_color": "Noir",
            "accepted_materials": [{"slug": "inconnu", "column": "XXX", "label": "Inconnu"}],
        },
        {
            "material_name": "Film",
            "bin_color": "Noir",
            "accepted_materials": [{"slug": "plastique", "column": "VER", "label": "Verre"}],
        },
        {"material_name": "", "bin_color": "Noir", "accepted_materials": None},
        ["not", "an", "object"],
    ],
)
def test_validate_response_body_rejects_invalid_bodies(body: Any) -> None:
    """Invalid bodies, including legacy fields and empty lists, fail validation.

    Args:
        body: Candidate response body.
    """
    assert validate_response_body(body) != []
