"""Local image dataset loading for the waste indexing pipeline."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Final

from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)


class ImageFetcher:
    """Load labelled images from a local dataset directory.

    The dataset is expected to contain one sub-directory per label, each
    holding the images associated with that label.
    """

    def __init__(
        self,
        root_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize the fetcher.

        Args:
            root_dir: Root directory of the local image dataset.
            semaphore: Concurrency limiter shared across image loads.
        """
        self._root_dir: Path = root_dir
        self._semaphore: asyncio.Semaphore = semaphore
        self._index: dict[str, list[Path]] | None = None

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize a label so it can be matched with a directory name.

        Args:
            label: Raw label or directory name.

        Returns:
            The label with non-alphanumeric characters replaced by
            underscores, in lower case.
        """
        return "".join(c if c.isalnum() else "_" for c in label).lower()

    @staticmethod
    def _list_images(label_dir: Path) -> list[Path]:
        """List the image files contained in a label directory.

        Args:
            label_dir: Directory holding the images of a single label.

        Returns:
            A sorted list of image paths, or an empty list on failure.
        """
        try:
            return sorted(
                path
                for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        except OSError as error:
            logger.warning(
                "Cannot read label directory '%s': %s", label_dir, error
            )
            return []

    def open_dir(self) -> dict[str, list[Path]]:
        """Open the root directory and map each label to its images.

        Args:
            None.

        Returns:
            A dictionary mapping each label (sub-directory name) to the
            sorted list of image paths it contains.

        Raises:
            FileNotFoundError: If the root directory does not exist.
            NotADirectoryError: If the root path is not a directory.
        """
        if not self._root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self._root_dir}"
            )
        if not self._root_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self._root_dir}"
            )

        try:
            label_dirs: list[Path] = sorted(
                path for path in self._root_dir.iterdir() if path.is_dir()
            )
        except OSError as error:
            logger.error(
                "Cannot read dataset directory '%s': %s",
                self._root_dir,
                error,
            )
            return {}

        index: dict[str, list[Path]] = {}
        for label_dir in label_dirs:
            images: list[Path] = self._list_images(label_dir)
            if not images:
                logger.warning("No image found for label '%s'.", label_dir.name)
                continue
            index[label_dir.name] = images

        logger.info(
            "Loaded %d label(s) and %d image(s) from '%s'.",
            len(index),
            sum(len(paths) for paths in index.values()),
            self._root_dir,
        )
        self._index = index
        return index

    def samples(self) -> list[tuple[Path, str]]:
        """Return every image of the dataset paired with its label.

        Args:
            None.

        Returns:
            A list of (image path, label) tuples.
        """
        index: dict[str, list[Path]] = (
            self._index if self._index is not None else self.open_dir()
        )
        return [
            (path, label) for label, paths in index.items() for path in paths
        ]

    def _resolve_paths(self, label: str) -> list[Path]:
        """Find the image paths associated with a label.

        Args:
            label: Waste label, either raw or matching a directory name.

        Returns:
            The image paths for the label, or an empty list if unknown.
        """
        index: dict[str, list[Path]] = (
            self._index if self._index is not None else self.open_dir()
        )
        if label in index:
            return index[label]
        target: str = self._normalize_label(label)
        for directory_name, paths in index.items():
            if self._normalize_label(directory_name) == target:
                return paths
        return []

    @staticmethod
    def _load_image(path: Path) -> Image.Image | None:
        """Open and decode a single image file.

        Args:
            path: Path of the image file.

        Returns:
            A decoded RGB PIL image, or None if the file cannot be read.
        """
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, Image.DecompressionBombError) as error:
            logger.warning("Cannot open image '%s': %s", path, error)
            return None

    async def _load_one(self, path: Path) -> Image.Image | None:
        """Load a single image without blocking the event loop.

        Args:
            path: Path of the image file.

        Returns:
            A decoded RGB PIL image, or None if loading fails.
        """
        async with self._semaphore:
            return await asyncio.to_thread(self._load_image, path)

    async def fetch(self, label: str, count: int) -> list[Image.Image]:
        """Load up to ``count`` local images for a label.

        Args:
            label: Waste label to load images for.
            count: Maximum number of images to load.

        Returns:
            A list of successfully decoded PIL images.
        """
        paths: list[Path] = (await asyncio.to_thread(self._resolve_paths, label))[
            :count
        ]
        if not paths:
            logger.warning("No local images available for '%s'.", label)
            return []

        loaded: list[Image.Image | None] = await asyncio.gather(
            *(self._load_one(path) for path in paths)
        )
        images: list[Image.Image] = [
            image for image in loaded if image is not None
        ]
        logger.info(
            "Loaded %d/%d image(s) for '%s'.", len(images), count, label
        )
        return images
