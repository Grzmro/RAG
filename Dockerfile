# syntax=docker/dockerfile:1.7

# --- build ------------------------------------------------------------------
# The uv image is python:3.13-slim-bookworm with uv on top, so the virtualenv
# built here keeps working when it is copied into the runtime stage.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

# PATH puts the project venv first, so `python` in the prewarm step below means
# the project's interpreter rather than the system one the base image ships.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the source and the project itself. The install stays editable on
# purpose: rag.config derives PROJECT_ROOT from the package location, and a
# non-editable install would resolve it to site-packages, breaking every
# relative DOCS_DIR/PERSIST_DIR and the .env lookup.
COPY README.md ./
COPY rag/ ./rag/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Bake the embedding weights into the image. A 2 GB box should not spend its
# first request downloading ~130 MB of ONNX, and the runtime stage then needs
# no network access at all beyond the Anthropic API.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBEDDING_MODEL'])"

# --- runtime ----------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# OMP_NUM_THREADS: t3.small is 2 burstable vCPUs; letting ONNX open a thread
# per core it thinks it sees just burns CPU credits on context switching.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    DOCS_DIR=/data/docs \
    PERSIST_DIR=/data/chroma \
    OMP_NUM_THREADS=2

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /data/docs /data/chroma \
    && chown -R rag:rag /data

WORKDIR /app
COPY --from=build --chown=rag:rag /opt/fastembed /opt/fastembed
COPY --from=build /app /app

USER rag
EXPOSE 8000

# /health is the only endpoint that touches neither the index nor the API key.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# One worker deliberately: Chroma persists through SQLite (single writer), and
# every extra worker would load its own copy of the embedding model.
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
