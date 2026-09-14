"""Controller layer bridging the API view and the model."""

from PIL import Image

from .model import Model, Response


class Controler:
    """Coordinates model access for the viewer layer."""

    model: Model

    def __init__(self) -> None:
        """Instantiate the underlying model wrapper."""
        self.model = Model()

    async def get_model_response(self, image: Image.Image, region: str) -> Response:
        """Delegate classification to the model asynchronously.

        Args:
            image: The PIL image to classify.

        Returns:
            The model Response for the given image.

        """
        return await self.model.predict_material(image, region)
