"""Entry point starting the Greener API server."""

from __future__ import annotations

import uvicorn

from configs import Settings, get_settings
from src.viewer import Viewer

app: Viewer = Viewer()


def run() -> None:
    """Start the Uvicorn server with the configured host and port."""
    settings: Settings = get_settings()
    uvicorn.run(app=app, host=settings.ai_host, port=settings.ai_port)


if __name__ == "__main__":
    run()