"""Download a file from Google Drive to a secure local destination."""
# ruff: noqa: INP001, CPY001

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

logger = logging.getLogger("download_gdrive")

ALLOWED_HOSTS: Final = frozenset({"drive.google.com", "docs.google.com"})
FILE_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{20,}$")
PATH_ID_PATTERN: Final = re.compile(r"/(?:file/)?d/([A-Za-z0-9_-]+)")
API_URL: Final = (
    "https://www.googleapis.com/drive/v3/files/{file_id}"
    "?alt=media&supportsAllDrives=true"
)
RETRYABLE_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
PERMANENT_HINTS: Final = {
    401: "token invalid or expired",
    403: "file not shared with this account or missing scope",
    404: "file not found, check the file ID",
}
MAX_ATTEMPTS: Final = 5
BASE_DELAY: Final = 2.0
CHUNK_SIZE: Final = 1024 * 1024
DEFAULT_MAX_SIZE: Final = 200 * 1024 * 1024
DEFAULT_TIMEOUT: Final = 30.0
FILE_MODE: Final = 0o640


class DownloadError(Exception):
    """Base error for every download failure."""


class InvalidUrlError(DownloadError):
    """Raised when the URL is not a valid Google Drive link."""


class AuthenticationError(DownloadError):
    """Raised when no usable access token is available."""


class UnsafePathError(DownloadError):
    """Raised when the destination path is not safe to write."""


class TransientError(DownloadError):
    """Raised for errors that may succeed on retry."""


class IntegrityError(DownloadError):
    """Raised when size or checksum validation fails."""


def extract_file_id(url: str) -> str:
    """Validate a Google Drive URL and return its file ID.

    Args:
        url: Shareable Google Drive link.

    Returns:
        The validated file ID.

    Raises:
        InvalidUrlError: If the URL is not a supported Drive link.
    """
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https":
        msg = "only HTTPS URLs are accepted"
        raise InvalidUrlError(msg)
    if parsed.hostname not in ALLOWED_HOSTS:
        msg = f"unsupported host: {parsed.hostname}"
        raise InvalidUrlError(msg)

    match = PATH_ID_PATTERN.search(parsed.path)
    if match:
        candidate = match.group(1)
    else:
        ids = urllib.parse.parse_qs(parsed.query).get("id", [])
        candidate = ids[0] if ids else ""

    if not FILE_ID_PATTERN.fullmatch(candidate):
        msg = "no valid file ID found in URL"
        raise InvalidUrlError(msg)
    return candidate


def load_token() -> str:
    """Read the OAuth access token from the environment.

    The token is read from ``GDRIVE_TOKEN`` or from the file referenced
    by ``GDRIVE_TOKEN_FILE``. It is never accepted as a CLI argument.

    Returns:
        The access token.

    Raises:
        AuthenticationError: If no token is available.
    """
    token = os.environ.get("GDRIVE_TOKEN", "").strip()
    token_file = os.environ.get("GDRIVE_TOKEN_FILE", "").strip()
    if not token and token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            msg = "cannot read token file"
            raise AuthenticationError(msg) from exc
    if not token:
        msg = "set GDRIVE_TOKEN or GDRIVE_TOKEN_FILE"
        raise AuthenticationError(msg)
    return token


def resolve_destination(
    destination: str, base_dir: Path, *, overwrite: bool
) -> Path:
    """Resolve the destination and ensure it stays inside base_dir.

    Args:
        destination: Requested output path.
        base_dir: Directory the output must remain within.
        overwrite: Whether an existing file may be replaced.

    Returns:
        The resolved, safe destination path.

    Raises:
        UnsafePathError: If the path escapes base_dir, is a symlink,
            or already exists without overwrite.
    """
    base = base_dir.resolve(strict=True)
    raw = Path(destination)
    candidate = raw if raw.is_absolute() else base / raw

    if candidate.is_symlink():
        msg = "destination is a symbolic link"
        raise UnsafePathError(msg)

    target = candidate.resolve()
    if not target.is_relative_to(base):
        msg = f"destination escapes base directory: {base}"
        raise UnsafePathError(msg)
    if target.is_dir():
        msg = "destination is a directory"
        raise UnsafePathError(msg)
    if target.exists() and not overwrite:
        msg = "destination exists, use --overwrite"
        raise UnsafePathError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _stream_to_file(
    request: urllib.request.Request,
    target: Path,
    max_size: int,
    timeout: float,
) -> tuple[Path, str]:
    """Perform one download attempt into a temporary file.

    Returns:
        The temporary file path and its SHA-256 hex digest.

    Raises:
        TransientError: On network errors or retryable HTTP statuses.
        DownloadError: On permanent HTTP errors or invalid content.
        IntegrityError: If the file exceeds max_size.
    """
    try:
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        if exc.code in RETRYABLE_STATUS:
            msg = f"HTTP {exc.code}"
            raise TransientError(msg) from exc
        hint = PERMANENT_HINTS.get(exc.code, "unexpected response")
        msg = f"HTTP {exc.code}: {hint}"
        raise DownloadError(msg) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        msg = f"network error: {exc}"
        raise TransientError(msg) from exc

    with response:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            msg = "received an HTML page instead of the file"
            raise DownloadError(msg)

        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_size:
            msg = f"file too large: {declared} bytes"
            raise IntegrityError(msg)

        digest = hashlib.sha256()
        written = 0
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".part"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_size:
                        msg = f"file exceeds {max_size} bytes"
                        raise IntegrityError(msg)
                    digest.update(chunk)
                    handle.write(chunk)
        except (TimeoutError, ConnectionError) as exc:
            tmp_path.unlink(missing_ok=True)
            msg = f"connection lost during download: {exc}"
            raise TransientError(msg) from exc
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    return tmp_path, digest.hexdigest()


def download_file(
    file_id: str,
    target: Path,
    token: str,
    *,
    expected_sha256: str | None,
    max_size: int,
    timeout: float,
) -> str:
    """Download a Drive file atomically with retries and validation.

    Args:
        file_id: Validated Google Drive file ID.
        target: Safe destination path.
        token: OAuth access token.
        expected_sha256: Optional expected SHA-256 hex digest.
        max_size: Maximum accepted size in bytes.
        timeout: Socket timeout in seconds.

    Returns:
        The SHA-256 hex digest of the downloaded file.

    Raises:
        DownloadError: If the download ultimately fails.
    """
    request = urllib.request.Request(  # noqa: S310
        API_URL.format(file_id=file_id),
        headers={"Authorization": f"Bearer {token}"},
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            tmp_path, sha256 = _stream_to_file(
                request, target, max_size, timeout
            )
            break
        except TransientError as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = BASE_DELAY * 2 ** (attempt - 1) + random.uniform(0, 1)  # noqa: S311
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)

    if expected_sha256 and sha256 != expected_sha256.lower():
        tmp_path.unlink(missing_ok=True)
        msg = f"checksum mismatch: expected {expected_sha256}, got {sha256}"
        raise IntegrityError(msg)

    tmp_path.chmod(FILE_MODE)
    tmp_path.replace(target)
    return sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, defaults to sys.argv.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download a Google Drive file securely."
    )
    parser.add_argument("url", help="Google Drive shareable link")
    parser.add_argument("destination", help="Local output path")
    parser.add_argument("--sha256", help="Expected SHA-256 hex digest")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the downloader.

    Args:
        argv: Argument list, defaults to sys.argv.

    Returns:
        Process exit code.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )
    args = parse_args(argv)
    try:
        file_id = extract_file_id(args.url)
        token = load_token()
        target = resolve_destination(
            args.destination, args.base_dir, overwrite=args.overwrite
        )
        sha256 = download_file(
            file_id,
            target,
            token,
            expected_sha256=args.sha256,
            max_size=args.max_size,
            timeout=args.timeout,
        )
    except DownloadError as exc:
        logger.error("%s", exc)  # noqa: TRY400
        return 1
    except FileNotFoundError:
        logger.error("base directory does not exist")  # noqa: TRY400
        return 1

    logger.info("Downloaded %s (sha256=%s)", target, sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
