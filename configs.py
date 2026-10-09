"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_MODEL_FILES: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
)


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment.

    Attributes:
        pipeline_model_name: Hugging Face model identifier used by the
            image-classification pipeline.
        host: Network interface the server binds to.
        port: TCP port the server listens on.
        embedding_model_name: Hugging Face model used for image embeddings.
        qdrant_host: Hostname of the Qdrant server.
        qdrant_port: Port of the Qdrant server.
        data_dir: Directory containing one JSON file per region.
        images_per_label: Number of reference images fetched per label.
        request_timeout: Timeout in seconds for image download requests.
        max_retries: Maximum download attempts per image.
        max_concurrent_downloads: Maximum simultaneous image downloads.
        device: Torch device used for embedding inference.
        save_images: Whether to persist downloaded images locally.
        image_backup_dir: Directory where images are stored as backup.
        log_level: Logging verbosity level.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    ai_host: str = Field(
        default="0.0.0.0",
        alias="AI_HOST",
        description="Server bind address.",
    )
    ai_port: int = Field(
        default=8000,
        alias="AI_PORT",
        ge=1,
        le=65535,
        description="Server listening port.",
    )
    embedding_model_name: str = Field(
        default="facebook/dinov2-large",
        alias="EMBEDDING_MODEL_NAME",
        description="Hugging Face embedding model name.",
    )
    qdrant_host: str = Field(
        default="localhost",
        alias="QDRANT_HOST",
        description="Qdrant server hostname.",
    )
    qdrant_port: int = Field(
        default=6333,
        alias="QDRANT_PORT",
        ge=1,
        le=65535,
        description="Qdrant server port.",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        alias="DATA_DIR",
        description="Directory holding one JSON file per region.",
    )

    device: str = Field(
        default="cpu",
        alias="DEVICE",
        description="Torch device for embedding inference.",
    )

    image_dir: Path = Field(
        default=Path("./image_dir"),
        alias="IMAGE_DIR",
        description="Directory for image backups.",
    )

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging verbosity level.",
    )

    max_concurrent_uploads: int = Field(
        default=10,
        alias="MAX_CONCURRENT_UPLOADS",
        description=" Max concurrent image upoads.",
    )
    images_per_label: int = Field(
        default=3,
        alias="IMAGES_PER_LABEL",
        description="number of images per label",
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_model_name: str = Field(
        default="facebook/dinov2-large",
        alias="EMBEDDING_MODEL_NAME",
        description="Hugging Face embedding model name.",
    )
    embedding_model_dir: Path = Field(
        default=Path("./models"),
        alias="EMBEDDING_MODEL_DIR",
        description="Root directory where embedding models are stored.",
    )

    @property
    def embedding_model_path(self) -> Path:
        """Return the local directory of the configured embedding model.

        Returns:
            The model directory, derived from the model name so that
            several models can coexist under the same root.

        """
        return self.embedding_model_dir / self.embedding_model_name.replace("/", "--")


def is_model_available(model_path: Path) -> bool:
    """Check whether a local model directory contains a loadable model.

    Args:
        model_path: Directory expected to contain the model files.

    Returns:
        True if the configuration and weight files are present.

    """
    model_path = Path(model_path)
    has_configs = all(((model_path / name).is_file() ) for name in REQUIRED_MODEL_FILES)
    has_weights = any(model_path.glob("*.safetensors"))
    return has_configs and has_weights



@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the settings.

    Using an LRU cache guarantees the environment is parsed once and the
    same immutable Settings object is reused across the application.

    Returns:
        The validated Settings instance.

    """
    return Settings()  # type: ignore
