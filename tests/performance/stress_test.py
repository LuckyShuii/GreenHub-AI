"""Run a bounded concurrent image-upload test against an authorized API."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

DEFAULT_URL = "https://greenhub.lucasboillot.fr/ai/greener/upload/dechets"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


@dataclass(frozen=True)
class Config:
    """Store upload settings and load-test limits."""

    url: str
    images: tuple[Path, ...]
    concurrency: int
    count: int
    timeout: float
    field: str
    token: str | None


@dataclass(frozen=True)
class Result:
    """Store the outcome and elapsed time of one upload."""

    number: int
    image: str
    status: int | None
    elapsed: float
    success: bool
    response: str


def validate_image(path: Path) -> str:
    """Validate image contents and return their detected MIME type."""
    if not path.is_file():
        raise ValueError(f"Fichier introuvable : {path}")

    size = path.stat().st_size
    if not 0 < size <= MAX_IMAGE_BYTES:
        raise ValueError(f"Image vide ou supérieure à 10 Mio : {path}")

    try:
        with Image.open(path) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise ValueError(f"Format non accepté : {path}")
            image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image invalide : {path}") from exc

    return Image.MIME[image_format]


def upload(
    number: int,
    path: Path,
    mime_type: str,
    config: Config,
) -> Result:
    """Send one upload without retries or automatic redirects."""
    started = time.perf_counter()
    status: int | None = None
    success = False
    headers = {"Accept": "application/json"}

    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    try:
        with path.open("rb") as image_file:
            with requests.post(
                config.url,
                files={
                    config.field: (path.name, image_file, mime_type),
                },
                headers=headers,
                timeout=(5.0, config.timeout),
                allow_redirects=False,
            ) as response:
                status = response.status_code
                success = 200 <= status < 300
                try:
                    body = json.dumps(response.json(), ensure_ascii=False)
                except ValueError:
                    body = response.text

                if status in {401, 403}:
                    body = f"Authentification ou accès refusé : {body}"
                elif status == 413:
                    body = f"Image trop volumineuse : {body}"
                elif status in {415, 422}:
                    body = f"Format ou données refusés : {body}"

    except requests.exceptions.Timeout:
        body = "Délai de connexion ou de lecture dépassé."
    except requests.exceptions.SSLError as exc:
        body = f"Erreur TLS : {exc}"
    except requests.exceptions.ConnectionError as exc:
        body = f"Erreur de connexion : {exc}"
    except requests.exceptions.RequestException as exc:
        body = f"Erreur HTTP : {exc}"
    except OSError as exc:
        body = f"Erreur de lecture du fichier : {exc}"

    return Result(
        number=number,
        image=path.name,
        status=status,
        elapsed=time.perf_counter() - started,
        success=success,
        response=body,
    )


def parse_args() -> Config:
    """Parse and validate command-line settings."""
    parser = argparse.ArgumentParser(
        description="Test concurrent et borné d'upload d'images.",
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--url",
        default=os.getenv("GREENER_API_URL", DEFAULT_URL),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--count",
        type=int,
        help="Nombre total de requêtes ; répète les images si nécessaire.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--field", default="file")
    args = parser.parse_args()

    url = urlsplit(args.url)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        parser.error("L'URL doit utiliser HTTPS, sans identifiants intégrés.")
    if not 1 <= args.concurrency <= 8:
        parser.error("--concurrency doit être compris entre 1 et 8.")
    if not 0 < args.timeout <= 120:
        parser.error("--timeout doit être compris entre 0 (exclu) et 120.")
    if not args.field.strip():
        parser.error("--field ne doit pas être vide.")

    count = args.count if args.count is not None else len(args.images)
    if not 1 <= count <= 100:
        parser.error("Le nombre total de requêtes doit être compris entre 1 et 100.")

    return Config(
        url=args.url,
        images=tuple(args.images),
        concurrency=args.concurrency,
        count=count,
        timeout=args.timeout,
        field=args.field,
        token=os.getenv("GREENER_API_TOKEN"),
    )


def main() -> int:
    """Validate inputs, execute uploads, and print a test summary."""
    config = parse_args()

    try:
        mime_types = {
            path: validate_image(path)
            for path in config.images
        }
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"Validation échouée : {exc}")
        return 1

    results: list[Result] = []
    started = time.perf_counter()
    stopped = False

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = []
        for index in range(config.count):
            path = config.images[index % len(config.images)]
            futures.append(
                executor.submit(
                    upload,
                    index + 1,
                    path,
                    mime_types[path],
                    config,
                ),
            )

        for future in as_completed(futures):
            if future.cancelled():
                continue

            result = future.result()
            results.append(result)
            label = "OK" if result.success else "ÉCHEC"
            print(
                f"[{len(results)}/{config.count}] "
                f"#{result.number} {result.image} | {label} | "
                f"HTTP {result.status or '-'} | {result.elapsed:.2f} s",
                flush=True,
            )
            print(f"Réponse : {result.response[:2000]}", flush=True)

            if result.status in {401, 403, 429, 503} and not stopped:
                stopped = True
                print(
                    "Accès refusé ou serveur indisponible : "
                    "annulation des requêtes encore en attente.",
                    flush=True,
                )
                for pending in futures:
                    pending.cancel()

    elapsed = time.perf_counter() - started
    successes = sum(result.success for result in results)
    average = sum(result.elapsed for result in results) / len(results)

    print("\nBilan")
    print(f"Requêtes terminées : {len(results)}/{config.count}")
    print(f"Succès : {successes}")
    print(f"Échecs : {len(results) - successes}")
    print(f"Durée totale : {elapsed:.2f} s")
    print(f"Latence moyenne côté client : {average:.2f} s")
    print(f"Débit observé : {len(results) / elapsed:.2f} requêtes/s")

    return 0 if successes == config.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
