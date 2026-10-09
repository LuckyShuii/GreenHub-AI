"""Download the embedding model to local storage for offline loading."""

import argparse
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from configs import get_settings, is_model_available
from logging_config import configure_logging, get_logger

logger = get_logger(__name__)

ALLOWED_PATTERNS: tuple[str, ...] = ("*.json", "*.safetensors", "*.txt")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="Download the embedding model defined by EMBEDDING_MODEL_NAME.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the model even if it is already present locally.",
    )
    return parser.parse_args()


def download_model(model_name: str, force: bool) -> int:
    """Download a Hugging Face model snapshot into the local model directory.

    Args:
        model_name: Hugging Face identifier of the model.
        force: Whether to download even if the model is already present.

    Returns:
        The process exit code, 0 on success and 1 on failure.

    """
    settings = get_settings()
    target_dir = settings.embedding_model_path

    if not force and is_model_available(target_dir):
        logger.info("Model '%s' already present in '%s'.", model_name, target_dir)
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model '%s' to '%s'.", model_name, target_dir)
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            allow_patterns=list(ALLOWED_PATTERNS),
        )
    except HfHubHTTPError as error:
        logger.error("Failed to download model '%s': %s", model_name, error)
        return 1

    if not is_model_available(target_dir):
        logger.error("Download finished but '%s' is incomplete.", target_dir)
        return 1

    logger.info("Model '%s' downloaded successfully.", model_name)
    return 0


def main() -> int:
    """Run the model download script.

    Returns:
        The process exit code.

    """
    arguments = parse_arguments()
    settings = get_settings()
    configure_logging(settings.log_level)
    return download_model(settings.embedding_model_name, arguments.force)


if __name__ == "__main__":
    sys.exit(main())
