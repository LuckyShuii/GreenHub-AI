"""Fixtures for the performance test suite."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

BASE_URL: str = os.getenv("PERF_BASE_URL", "http://127.0.0.1:8000")
USE_LIVE_SERVER: bool = os.getenv("PERF_LIVE", "0") == "1"


@pytest_asyncio.fixture
async def performance_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide an HTTP client targeting the application under test.

    Yields:
        An httpx.AsyncClient bound either to a live server or to the
        in-process ASGI application with a stubbed embedder.
    """
    if USE_LIVE_SERVER:
        async with httpx.AsyncClient(
            base_url=BASE_URL, timeout=30.0
        ) as client:
            yield client
        return

    from unittest.mock import AsyncMock, patch

    from main import servapp

    prediction = AsyncMock(
        return_value={
            "material_name": "bouteille en verre",
            "bin_color": "verte",
            "score": 0.91,
        }
    )
    with patch("src.controller.Controller.get_model_response", prediction):
        transport = httpx.ASGITransport(app=servapp)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30.0
        ) as client:
            yield client