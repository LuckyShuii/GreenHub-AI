"""Unit tests for the resilient image fetcher."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from src.fetcher import ImageFetcher


def _png_bytes(color: tuple[int, int, int] = (10, 10, 10)) -> bytes:
    """Encode a tiny PNG image.

    Args:
        color: RGB fill color.

    Returns:
        PNG-encoded bytes.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fetcher(tmp_path: Path) -> ImageFetcher:
    """Build a fetcher writing backups into a temporary directory.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        A configured ImageFetcher instance.
    """
    return ImageFetcher(
        timeout=1,
        max_retries=3,
        semaphore=asyncio.Semaphore(4),
        save_images=False,
        backup_dir=tmp_path / "backup",
    )


class TestSearchUrls:
    """Behaviour of the DuckDuckGo search wrapper."""

    def test_appends_waste_keyword_to_query(
        self, fetcher: ImageFetcher
    ) -> None:
        """The search query is suffixed to bias results toward waste.

        Args:
            fetcher: Fetcher under test.
        """
        ddgs_instance = MagicMock()
        ddgs_instance.images.return_value = [
            {"image": "https://example.test/a.jpg"}
        ]
        context = MagicMock()
        context.__enter__.return_value = ddgs_instance

        with (
            patch("src.fetcher.DDGS", return_value=context),
            patch("src.fetcher.time.sleep"),
        ):
            urls = fetcher._search_urls("carton", 1)

        ddgs_instance.images.assert_called_once_with(
            "carton dechet", max_results=1
        )
        assert urls == ["https://example.test/a.jpg"]

    def test_returns_empty_list_on_search_failure(
        self, fetcher: ImageFetcher
    ) -> None:
        """Search backend errors degrade to an empty result list.

        Args:
            fetcher: Fetcher under test.
        """
        with (
            patch("src.fetcher.DDGS", side_effect=RuntimeError("rate limit")),
            patch("src.fetcher.time.sleep"),
        ):
            assert fetcher._search_urls("carton", 3) == []


class TestDownloadOne:
    """Retry and decoding behaviour of single-image downloads."""

    @pytest.mark.asyncio
    async def test_returns_decoded_image_on_success(
        self, fetcher: ImageFetcher
    ) -> None:
        """A 200 response with valid bytes yields an RGB image.

        Args:
            fetcher: Fetcher under test.
        """
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=_png_bytes())
        )
        async with httpx.AsyncClient(transport=transport) as client:
            image = await fetcher._download_one(
                client, "https://example.test/a.png"
            )

        assert image is not None
        assert image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_retries_until_success(
        self, fetcher: ImageFetcher
    ) -> None:
        """Transient failures are retried up to the configured limit.

        Args:
            fetcher: Fetcher under test.
        """
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(503)
            return httpx.Response(200, content=_png_bytes())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            image = await fetcher._download_one(
                client, "https://example.test/a.png"
            )

        assert image is not None
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_returns_none_after_exhausting_retries(
        self, fetcher: ImageFetcher
    ) -> None:
        """Persistent failures resolve to None rather than raising.

        Args:
            fetcher: Fetcher under test.
        """
        transport = httpx.MockTransport(
            lambda request: httpx.Response(404)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            image = await fetcher._download_one(
                client, "https://example.test/missing.png"
            )

        assert image is None

    @pytest.mark.asyncio
    async def test_returns_none_for_undecodable_payload(
        self, fetcher: ImageFetcher
    ) -> None:
        """Non-image bytes are handled without propagating exceptions.

        Args:
            fetcher: Fetcher under test.
        """
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"garbage")
        )
        async with httpx.AsyncClient(transport=transport) as client:
            image = await fetcher._download_one(
                client, "https://example.test/a.png"
            )

        assert image is None

    @pytest.mark.asyncio
    async def test_respects_concurrency_semaphore(
        self, tmp_path: Path
    ) -> None:
        """In-flight downloads never exceed the semaphore capacity.

        Args:
            tmp_path: Temporary directory fixture.
        """
        limit = 2
        current = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.01)
            current -= 1
            return httpx.Response(200, content=_png_bytes())

        fetcher = ImageFetcher(
            timeout=1,
            max_retries=1,
            semaphore=asyncio.Semaphore(limit),
            save_images=False,
            backup_dir=tmp_path,
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await asyncio.gather(
                *(
                    fetcher._download_one(
                        client, f"https://example.test/{index}.png"
                    )
                    for index in range(10)
                )
            )

        assert peak <= limit


class TestPersist:
    """Local backup behaviour."""

    def test_writes_sanitized_directory(
        self, fetcher: ImageFetcher, sample_image: Image.Image
    ) -> None:
        """Non-alphanumeric label characters become underscores.

        Args:
            fetcher: Fetcher under test.
            sample_image: Image to persist.
        """
        fetcher._persist(sample_image, "bouteille en verre", 0)

        expected = fetcher._backup_dir / "bouteille_en_verre" / "0.jpg"
        assert expected.exists()

    def test_creates_nested_directories(
        self, fetcher: ImageFetcher, sample_image: Image.Image
    ) -> None:
        """Missing parent directories are created on demand.

        Args:
            fetcher: Fetcher under test.
            sample_image: Image to persist.
        """
        fetcher._persist(sample_image, "carton", 4)
        assert (fetcher._backup_dir / "carton").is_dir()


class TestFetch:
    """End-to-end behaviour of the public fetch method."""

    @pytest.mark.asyncio
    async def test_filters_out_failed_downloads(
        self, fetcher: ImageFetcher
    ) -> None:
        """Only successfully decoded images are returned.

        Args:
            fetcher: Fetcher under test.
        """
        urls = [f"https://example.test/{index}.png" for index in range(4)]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(("0.png", "2.png")):
                return httpx.Response(200, content=_png_bytes())
            return httpx.Response(500)

        with patch.object(fetcher, "_search_urls", return_value=urls):
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                images = await fetcher.fetch(client, "carton", 4)

        assert len(images) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_search_yields_nothing(
        self, fetcher: ImageFetcher
    ) -> None:
        """No search results produce no images and no requests.

        Args:
            fetcher: Fetcher under test.
        """
        with patch.object(fetcher, "_search_urls", return_value=[]):
            transport = httpx.MockTransport(
                lambda request: httpx.Response(200, content=_png_bytes())
            )
            async with httpx.AsyncClient(transport=transport) as client:
                images = await fetcher.fetch(client, "inconnu", 3)

        assert images == []