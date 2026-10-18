"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_MODEL_FILES: tuple[str, ...] = ("config.json", "preprocessor_config.json")


class Settings(BaseSettings):
    """Strongly typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ai_host: str = Field(default="0.0.0.0", alias="AI_HOST")
    ai_port: int = Field(default=8000, alias="AI_PORT", ge=1, le=65535)
    embedding_model_name: str = Field(
        default="facebook/dinov2-large", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_model_dir: Path = Field(
        default=Path("./models"), alias="EMBEDDING_MODEL_DIR"
    )
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT", ge=1, le=65535)
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    device: str = Field(default="cpu", alias="DEVICE")
    image_dir: Path = Field(default=Path("./image_dir"), alias="IMAGE_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_concurrent_uploads: int = Field(
        default=2,
        alias="MAX_CONCURRENT_UPLOADS",
        ge=1,
        description="Maximum number of concurrent embedding operations.",
    )
    max_upload_size: int = Field(
        default=10 * 1024 * 1024,
        alias="MAX_UPLOAD_SIZE",
        ge=1,
        description="Maximum uploaded image size in bytes.",
    )
    images_per_label: int = Field(default=3, alias="IMAGES_PER_LABEL", ge=1)

    @property
    def embedding_model_path(self) -> Path:
        """Return the local directory of the configured embedding model."""
        return self.embedding_model_dir / self.embedding_model_name.replace("/", "--")


def is_model_available(model_path: Path) -> bool:
    """Check whether a local model directory contains a loadable model."""
    model_path = Path(model_path)
    has_configs = all((model_path / n).is_file() for n in REQUIRED_MODEL_FILES)
    has_weights = any(model_path.glob("*.safetensors"))
    return has_configs and has_weights


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()  # type: ignore
