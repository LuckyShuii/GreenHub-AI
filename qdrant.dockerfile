ARG QDRANT_VERSION=v1.18.2

FROM qdrant/qdrant:${QDRANT_VERSION}

ARG QDRANT_VERSION
ENV QDRANT_VERSION=${QDRANT_VERSION}

ENV QDRANT_PORT=${QDRANT_PORT} \
    QDRANT_GRPC_PORT=${QDRANT_GRPC_PORT} \
    QDRANT_STORAGE_PATH=/qdrant/storage \
    QDRANT__LOG_LEVEL=${QDRANT_LOG_LEVEL}\
    QDRANT_HOST=${QDRANT_HOST}

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
EXPOSE ${QDRANT_PORT}
EXPOSE ${QDRANT_GRPC_PORT}


HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -f http://${QDRANT_HOST}:${QDRANT_PORT}/healthz || exit 1

WORKDIR /qdrant

