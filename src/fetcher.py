"""Asynchronous image fetching with local backup support."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path

import httpx
from ddgs import DDGS
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageFetcher:
    """Fetch images from the web with retries and optional local backup."""

    def __init__(
        self,
        timeout: int,
        max_retries: int,
        semaphore: asyncio.Semaphore,
        save_images: bool,
        backup_dir: Path,
    ) -> None:
        """Initialize the fetcher.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum download attempts per image.
            semaphore: Concurrency limiter shared across downloads.
            save_images: Whether to persist images locally.
            backup_dir: Root directory for image backups.
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._semaphore = semaphore
        self._save_images = save_images
        self._backup_dir = backup_dir

    def _search_urls(self, query: str, count: int) -> list[str]:
        """Search image URLs for a given query.

        Args:
            query: Search term describing the waste item.
            count: Number of image URLs to retrieve.

        Returns:
            A list of image URLs.
        """
        try:
            with DDGS() as ddgs:
                time.sleep(5)
                results = ddgs.images(query + " dechet", max_results=count)
            return [item["image"] for item in results]
        except Exception as error:  # noqa: BLE001
            logger.warning("Image search failed for '%s': %s", query, error)
            return []

    async def _download_one(
        self, client: httpx.AsyncClient, url: str
    ) -> Image.Image | None:
        """Download and decode a single image with retries.

        Args:
            client: Shared asynchronous HTTP client.
            url: Image URL to download.

        Returns:
            A decoded PIL image, or None if all attempts fail.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._semaphore:
                    response = await client.get(url, timeout=self._timeout)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                return image
            except (httpx.HTTPError, UnidentifiedImageError, OSError) as error:
                logger.debug(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_retries,
                    url,
                    error,
                )
        logger.warning("All download attempts failed for %s", url)
        return None

    def _persist(self, image: Image.Image, label: str, index: int) -> None:
        """Persist an image locally as a backup.

        Args:
            image: Decoded PIL image to save.
            label: Waste label used to build the directory name.
            index: Positional index of the image within the label.
        """
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        target_dir = self._backup_dir / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
        image.save(target_dir / f"{index}.jpg", format="JPEG")

    async def fetch(
        self, client: httpx.AsyncClient, label: str, count: int
    ) -> list[Image.Image]:
        """Fetch and optionally persist images for a label.

        Args:
            client: Shared asynchronous HTTP client.
            label: Waste label to search images for.
            count: Number of images to fetch.

        Returns:
            A list of successfully downloaded PIL images.
        """
        urls = await asyncio.to_thread(self._search_urls, label, count)
        tasks = [self._download_one(client, url) for url in urls]
        downloaded = await asyncio.gather(*tasks)

        images: list[Image.Image] = []
        for position, image in enumerate(downloaded):
            if image is None:
                continue
            if self._save_images:
                try:
                    await asyncio.to_thread(self._persist, image, label, position)
                except OSError as error:
                    logger.warning("Failed to persist image for %s: %s", label, error)
            images.append(image)

        logger.info("Fetched %d/%d images for '%s'.", len(images), count, label)
        return images
