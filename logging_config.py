"""Centralized logging configuration for the application."""

import logging
import sys
from logging import getLogger


def configure_logging(log_level: str) -> None:
    """Configure the root logger with a consistent format.

    Args:
        log_level: Desired logging verbosity (e.g. INFO, DEBUG).

    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    )
    root = getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance.

    Args:
        name: Name of the logger, typically the module name.

    Returns:
        A configured Logger instance.

    """
    return getLogger(name)

