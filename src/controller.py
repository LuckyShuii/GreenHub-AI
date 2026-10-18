"""Controller layer bridging the API view and the model."""

from PIL import Image

from .embedder import ImageEmbedder
from .model import Model, Response


class Controller:
    """Coordinates model access for the viewer layer."""

    model: Model

    def __init__(self, embedder: ImageEmbedder | None = None) -> None:
        self.model = Model(embedder)

    def set_embedder(self, embedder: ImageEmbedder) -> None:
        self.model.set_embedder(embedder)

    async def get_model_response(self, image: Image.Image, region: str) -> Response:
        return await self.model.predict_material(image, region)
