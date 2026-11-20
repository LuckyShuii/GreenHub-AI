FROM python:3.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /greener
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

COPY pyproject.toml uv.lock ./
COPY main.py configs.py logging_config.py ./

COPY ./src ./src
COPY ./data ./data

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/greener/.venv/bin:$PATH" \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user: a compromise of the API must not grant root in the container.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin greener
USER greener
ENTRYPOINT ["python", "main.py"]
