FROM python:3.13-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /greener
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

COPY pyproject.toml uv.lock ./
COPY ./src ./src
COPY ./data ./data
COPY main.py configs.py logging_config.py download_gdrive.py  ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/greener/.venv/bin:$PATH"
RUN  download_gdrive.py "https://drive.google.com/file/d/1VuH2IHp0lqivDrCTyw7hUcUPawtd0JDd/view?usp=drive_link" ./image_dir.tar \
      --sha256 "$DATASET_SHA256"
RUN tar -xzf ./image_dir.tar -C ./ --no-same-owner --no-same-permissions
RUN rm ./image_dir.tar
ENTRYPOINT ["python", "main.py"]
