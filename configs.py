"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default="facebook/dinov2-small",
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
    images_per_label: int = Field(
        default=5,
        alias="IMAGES_PER_LABEL",
        ge=1,
        description="Number of images fetched per label.",
    )
    request_timeout: int = Field(
        default=10,
        alias="REQUEST_TIMEOUT",
        ge=1,
        description="Image download timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        alias="MAX_RETRIES",
        ge=1,
        description="Maximum download attempts per image.",
    )
    max_concurrent_downloads: int = Field(
        default=10,
        alias="MAX_CONCURRENT_DOWNLOADS",
        ge=1,
        description="Maximum simultaneous downloads.",
    )
    device: str = Field(
        default="cpu",
        alias="DEVICE",
        description="Torch device for embedding inference.",
    )
    save_images: bool = Field(
        default=True,
        alias="SAVE_IMAGES",
        description="Persist downloaded images locally.",
    )
    image_backup_dir: Path = Field(
        default=Path("./image_backup"),
        alias="IMAGE_BACKUP_DIR",
        description="Directory for image backups.",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging verbosity level.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the settings.

    Using an LRU cache guarantees the environment is parsed once and the
    same immutable Settings object is reused across the application.

    Returns:
        The validated Settings instance.

    """
    return Settings()  # type: ignore
