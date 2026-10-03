"""Unit tests for the local dataset image fetcher."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from src.fetcher import ImageFetcher


def _write_image(path: Path, mode: str = "RGB") -> None:
    """Create a small valid image at a path.

    Args:
        path: Destination image path.
        mode: Pillow image mode to create.

    """
    color: int | tuple[int, int, int] = 1 if mode == "L" else (1, 2, 3)
    image: Image.Image = Image.new(mode, (4, 4), color=color)
    image.save(path)


def _fetcher(root_dir: Path) -> ImageFetcher:
    """Create a fetcher for a dataset root.

    Args:
        root_dir: Dataset root directory.

    Returns:
        A configured image fetcher.

    """
    return ImageFetcher(root_dir, asyncio.Semaphore(4))


class TestOpenDir:
    """Tests for dataset discovery and indexing."""

    def test_indexes_sorted_images_and_ignores_non_images(
        self, tmp_path: Path
    ) -> None:
        """Discovery sorts labels and files and filters unsupported entries.

        Args:
            tmp_path: Temporary directory fixture.

        """
        first: Path = tmp_path / "alpha"
        second: Path = tmp_path / "zeta"
        first.mkdir()
        second.mkdir()
        _write_image(first / "b.PNG")
        _write_image(first / "a.jpg")
        _write_image(second / "c.tIfF")
        (first / "notes.txt").write_text("not an image")
        (tmp_path / "root.png").write_bytes(b"ignored")

        result: dict[str, list[Path]] = _fetcher(tmp_path).open_dir()

        assert list(result) == ["alpha", "zeta"]
        assert result["alpha"] == [first / "a.jpg", first / "b.PNG"]
        assert result["zeta"] == [second / "c.tIfF"]

    def test_samples_associate_each_image_with_label(
        self, tmp_path: Path
    ) -> None:
        """Samples pair every discovered image with its directory label.

        Args:
            tmp_path: Temporary directory fixture.

        """
        label: Path = tmp_path / "paper"
        label.mkdir()
        _write_image(label / "one.png")
        _write_image(label / "two.webp")

        assert _fetcher(tmp_path).samples() == [
            (label / "one.png", "paper"),
            (label / "two.webp", "paper"),
        ]

    def test_open_dir_caches_index(self, tmp_path: Path) -> None:
        """Subsequent reads use the existing index instead of rescanning.

        Args:
            tmp_path: Temporary directory fixture.

        """
        label: Path = tmp_path / "paper"
        label.mkdir()
        _write_image(label / "one.png")
        fetcher: ImageFetcher = _fetcher(tmp_path)
        first: dict[str, list[Path]] = fetcher.open_dir()
        _write_image(label / "two.png")

        assert fetcher.samples() == [(label / "one.png", "paper")]
        assert fetcher.open_dir() is not first
        assert len(fetcher.samples()) == 2

    def test_empty_label_is_ignored_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty label directories are skipped and logged.

        Args:
            tmp_path: Temporary directory fixture.
            caplog: Log capture fixture.

        """
        (tmp_path / "empty").mkdir()

        assert _fetcher(tmp_path).open_dir() == {}
        assert "No image found for label 'empty'." in caplog.text

    def test_root_without_label_directories_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """A root containing no subdirectories has an empty index.

        Args:
            tmp_path: Temporary directory fixture.

        """
        (tmp_path / "file.txt").write_text("ignored")

        assert _fetcher(tmp_path).open_dir() == {}

    def test_root_iterdir_error_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A root listing permission error is converted to an empty index.

        Args:
            tmp_path: Temporary directory fixture.
            monkeypatch: Pytest monkeypatch fixture.

        """

        def raise_permission_error(path: Path) -> Iterator[Path]:
            """Raise a permission error for a patched directory listing.

            Args:
                path: Directory whose contents would be listed.

            Returns:
                Never returns normally.

            """
            raise PermissionError(path)

        monkeypatch.setattr(Path, "iterdir", raise_permission_error)

        assert _fetcher(tmp_path).open_dir() == {}

    def test_label_iterdir_error_skips_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single unreadable label is omitted from the index.

        Args:
            tmp_path: Temporary directory fixture.
            monkeypatch: Pytest monkeypatch fixture.

        """
        good: Path = tmp_path / "good"
        bad: Path = tmp_path / "bad"
        good.mkdir()
        bad.mkdir()
        _write_image(good / "good.png")
        original = Path.iterdir

        def list_or_raise(path: Path) -> Iterator[Path]:
            """Raise only while listing the selected label directory.

            Args:
                path: Directory whose contents would be listed.

            Returns:
                The original directory iterator for other paths.

            """
            if path == bad:
                raise PermissionError(path)
            return original(path)

        monkeypatch.setattr(Path, "iterdir", list_or_raise)

        assert list(_fetcher(tmp_path).open_dir()) == ["good"]

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        """A missing dataset root raises FileNotFoundError.

        Args:
            tmp_path: Temporary directory fixture.

        """
        with pytest.raises(FileNotFoundError):
            _fetcher(tmp_path / "missing").open_dir()

    def test_file_root_raises(self, tmp_path: Path) -> None:
        """A file used as root raises NotADirectoryError.

        Args:
            tmp_path: Temporary directory fixture.

        """
        root: Path = tmp_path / "root"
        root.write_text("not a directory")

        with pytest.raises(NotADirectoryError):
            _fetcher(root).open_dir()


class TestResolvePaths:
    """Tests for label resolution."""

    def test_normalizes_label_for_matching(self, tmp_path: Path) -> None:
        """Punctuation and case differences still resolve a label.

        Args:
            tmp_path: Temporary directory fixture.

        """
        label: Path = tmp_path / "Glass Bottle"
        label.mkdir()
        _write_image(label / "one.png")
        fetcher: ImageFetcher = _fetcher(tmp_path)

        assert fetcher._resolve_paths("glass-bottle") == [label / "one.png"]

    def test_unknown_label_returns_empty(self, tmp_path: Path) -> None:
        """Unknown labels resolve to no paths.

        Args:
            tmp_path: Temporary directory fixture.

        """
        assert _fetcher(tmp_path)._resolve_paths("unknown") == []


class TestLoadImage:
    """Tests for synchronous image decoding."""

    def test_corrupt_image_returns_none(self, tmp_path: Path) -> None:
        """Corrupt image bytes are handled without raising.

        Args:
            tmp_path: Temporary directory fixture.

        """
        path: Path = tmp_path / "corrupt.png"
        path.write_bytes(b"invalid image")

        assert ImageFetcher._load_image(path) is None

    def test_decompression_bomb_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pillow decompression bomb errors are handled without raising.

        Args:
            tmp_path: Temporary directory fixture.
            monkeypatch: Pytest monkeypatch fixture.

        """
        path: Path = tmp_path / "bomb.png"
        path.write_bytes(b"placeholder")
        open_mock: Mock = Mock(
            side_effect=Image.DecompressionBombError("bomb")
        )
        monkeypatch.setattr("src.fetcher.Image.open", open_mock)

        assert ImageFetcher._load_image(path) is None

    def test_load_image_converts_to_rgb(self, tmp_path: Path) -> None:
        """Images in another mode are returned as RGB.

        Args:
            tmp_path: Temporary directory fixture.

        """
        path: Path = tmp_path / "gray.png"
        _write_image(path, mode="L")

        image: Image.Image | None = ImageFetcher._load_image(path)

        assert image is not None
        assert image.mode == "RGB"


class TestFetch:
    """Tests for asynchronous image loading."""

    @pytest.mark.asyncio
    async def test_fetch_returns_valid_images_and_applies_count(
        self, tmp_path: Path
    ) -> None:
        """Fetch limits paths to count before loading them.

        Args:
            tmp_path: Temporary directory fixture.

        """
        label: Path = tmp_path / "photos"
        label.mkdir()
        _write_image(label / "a.png")
        _write_image(label / "b.png")
        (label / "c.png").write_bytes(b"bad")

        images: list[Image.Image] = await _fetcher(tmp_path).fetch(
            "photos", 2
        )

        assert len(images) == 2
        assert all(image.mode == "RGB" for image in images)

    @pytest.mark.asyncio
    async def test_fetch_returns_only_valid_images_after_corruption(
        self, tmp_path: Path
    ) -> None:
        """Fetch returns valid images when one selected file is corrupt.

        Args:
            tmp_path: Temporary directory fixture.

        """
        label: Path = tmp_path / "photos"
        label.mkdir()
        _write_image(label / "a.png")
        (label / "b.png").write_bytes(b"bad")

        images: list[Image.Image] = await _fetcher(tmp_path).fetch(
            "photos", 5
        )

        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_fetch_unknown_label_returns_empty_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown label produces no images and a warning.

        Args:
            tmp_path: Temporary directory fixture.
            caplog: Log capture fixture.

        """
        assert await _fetcher(tmp_path).fetch("unknown", 2) == []
        assert "No local images available for 'unknown'." in caplog.text
