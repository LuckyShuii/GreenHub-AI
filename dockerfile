FROM python:3.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /greener
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

COPY pyproject.toml uv.lock ./
COPY ./src ./src
COPY ./data ./data
COPY main.py configs.py logging_config.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/greener/.venv/bin:$PATH"

ENTRYPOINT ["python", "main.py"]
