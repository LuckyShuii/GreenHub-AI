"""Throughput benchmark for the waste classification endpoint."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from tqdm import tqdm

UPLOAD_COUNT: int = 10
REQUESTS_PER_UPLOAD: int = 20
MIN_REQUESTS_PER_SECOND: float = 50.0
CONCURRENCY: int = 16
TARGET_REGION: str = "occitanie"
ENDPOINT_PATH: str = "/greener/upload/dechets"
IMAGE_PATH: Path = Path(__file__).parents[1] / "data" / "test.jpeg"
ACCEPTED_STATUS: frozenset[int] = frozenset({200, 404, 422})


@dataclass
class BenchmarkReport:
    """Aggregated timing results of a benchmark run."""

    latencies: list[float] = field(default_factory=list)
    failures: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def total_requests(self) -> int:
        """Return the number of completed requests.

        Returns:
            Count of recorded latencies.
        """
        return len(self.latencies)

    @property
    def wall_time(self) -> float:
        """Return the elapsed wall-clock duration in seconds.

        Returns:
            Duration of the run, never zero.
        """
        return max(self.finished_at - self.started_at, 1e-9)

    @property
    def throughput(self) -> float:
        """Return the achieved request rate.

        Returns:
            Requests per second over the whole run.
        """
        return self.total_requests / self.wall_time

    @property
    def mean_latency_ms(self) -> float:
        """Return the average latency in milliseconds.

        Returns:
            Mean latency, or zero when no request completed.
        """
        if not self.latencies:
            return 0.0
        return statistics.fmean(self.latencies) * 1000.0

    @property
    def p95_latency_ms(self) -> float:
        """Return the 95th percentile latency in milliseconds.

        Returns:
            Percentile latency, or zero when no request completed.
        """
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index] * 1000.0

    def summary(self) -> str:
        """Render a human-readable summary of the run.

        Returns:
            A multi-line report string.
        """
        return (
            f"requests={self.total_requests} "
            f"failures={self.failures} "
            f"wall_time={self.wall_time:.2f}s "
            f"throughput={self.throughput:.1f} req/s "
            f"mean={self.mean_latency_ms:.1f}ms "
            f"p95={self.p95_latency_ms:.1f}ms"
        )


def load_test_image() -> bytes:
    """Read the benchmark image from the data directory.

    Returns:
        Raw PNG bytes.

    Raises:
        pytest.skip.Exception: If the image is missing.
    """
    if not IMAGE_PATH.is_file():
        pytest.skip(f"benchmark image not found: {IMAGE_PATH}")
    return IMAGE_PATH.read_bytes()


async def issue_request(
    client: httpx.AsyncClient,
    payload: bytes,
    report: BenchmarkReport,
    progress: tqdm,
    guard: asyncio.Semaphore,
) -> None:
    """Send a single classification request and record its latency.

    Args:
        client: Shared HTTP client.
        payload: Raw image bytes to upload.
        report: Accumulator for timings and failures.
        progress: Progress bar updated after completion.
        guard: Semaphore bounding in-flight requests.
    """
    async with guard:
        files = {"file": ("test.png", payload, "image/png")}
        data = {"region": TARGET_REGION}
        start = time.perf_counter()
        try:
            response = await client.post(
                ENDPOINT_PATH, files=files, data=data
            )
            elapsed = time.perf_counter() - start
            report.latencies.append(elapsed)
            if response.status_code not in ACCEPTED_STATUS:
                report.failures += 1
        except httpx.HTTPError:
            report.latencies.append(time.perf_counter() - start)
            report.failures += 1
        progress.update(1)
        if report.latencies:
            progress.set_postfix(
                rps=f"{len(report.latencies) / max(time.perf_counter() - report.started_at, 1e-9):.1f}",
                mean=f"{report.mean_latency_ms:.0f}ms",
                err=report.failures,
            )


async def run_benchmark(
    client: httpx.AsyncClient,
    payload: bytes,
) -> BenchmarkReport:
    """Execute the full upload and request matrix.

    Args:
        client: Shared HTTP client.
        payload: Raw image bytes reused for every request.

    Returns:
        The populated benchmark report.
    """
    total = UPLOAD_COUNT * REQUESTS_PER_UPLOAD
    report = BenchmarkReport()
    guard = asyncio.Semaphore(CONCURRENCY)

    with tqdm(total=total, desc="classification", unit="req") as progress:
        report.started_at = time.perf_counter()
        for _ in range(UPLOAD_COUNT):
            tasks = [
                issue_request(client, payload, report, progress, guard)
                for _ in range(REQUESTS_PER_UPLOAD)
            ]
            await asyncio.gather(*tasks)
        report.finished_at = time.perf_counter()

    return report


@pytest.mark.performance
@pytest.mark.asyncio
async def test_endpoint_sustains_target_throughput(
    performance_client: httpx.AsyncClient,
) -> None:
    """The endpoint sustains the configured minimum request rate.

    Args:
        performance_client: HTTP client bound to the running application.
    """
    payload = load_test_image()

    report = await run_benchmark(performance_client, payload)

    print(f"\n{report.summary()}")
    assert report.total_requests == UPLOAD_COUNT * REQUESTS_PER_UPLOAD
    assert report.failures == 0, f"unexpected failures: {report.failures}"
    assert report.throughput >= MIN_REQUESTS_PER_SECOND, (
        f"throughput {report.throughput:.1f} req/s below target "
        f"{MIN_REQUESTS_PER_SECOND:.1f} req/s")

