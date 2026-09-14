"""Asynchronous image embedding using DINOv2."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """Encode images into dense vectors using a DINOv2 model.

    Attributes:
        dimension: Dimensionality of the produced embedding vectors.
    """

    def __init__(self, model_name: str, device: str) -> None:
        """Initialize the embedder with a pretrained model.

        Args:
            model_name: Hugging Face identifier of the model.
            device: Torch device used for inference.
        """
        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()
        self.dimension: int = self._model.config.hidden_size

    def _embed_sync(self, image: Image.Image) -> list[float]:
        """Synchronously embed a single image.

        Args:
            image: PIL image to encode.

        Returns:
            The embedding vector as a list of floats.
        """
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze(0)
        normalized = vector / vector.norm(p=2)
        return normalized.cpu().to(torch.float32).numpy().astype(np.float32).tolist()

    async def embed(self, image: Image.Image) -> list[float]:
        """Asynchronously embed a single image without blocking the loop.

        Args:
            image: PIL image to encode.

        Returns:
            The embedding vector as a list of floats.
        """
        return await asyncio.to_thread(self._embed_sync, image)
