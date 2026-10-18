"""Asynchronous image embedding using DINOv2."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from configs import is_model_available

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """Encode images into dense vectors using a locally stored DINOv2 model."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if not is_model_available(model_path):
            message = (
                f"Embedding model not found in '{model_path}'. "
                "Run 'python download_model.py' first."
            )
            raise FileNotFoundError(message)

        logger.info("Loading embedding model from '%s'.", model_path)
        self._device = device
        self._semaphore = semaphore or asyncio.Semaphore(1)
        self._processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            model_path, local_files_only=True
        ).to(device)
        self._model.eval()
        self.dimension: int = self._model.config.hidden_size

    def _embed_sync(self, image: Image.Image) -> list[float]:
        """Synchronously embed a single image."""
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze(0)
        normalized = vector / vector.norm(p=2)
        return normalized.cpu().to(torch.float32).numpy().astype(np.float32).tolist()

    async def embed(self, image: Image.Image) -> list[float]:
        """Embed an image without blocking the event loop."""
        async with self._semaphore:
            return await asyncio.to_thread(self._embed_sync, image)
