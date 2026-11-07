# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS frontend-builder
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
# Public build setting only. A slash uses the API on the same origin.
ARG VITE_API_BASE_URL=/
RUN VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run build

FROM python:3.12-slim-bookworm AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM python-base AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md requirements-runtime.txt ./
COPY src/ ./src/
# Pinned Python 3.12 Linux x86_64 dependencies include CPU-only PyTorch.
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && python -m pip install --no-cache-dir --no-deps . \
    && python -m pip check

FROM python-base AS runtime

# Required by the CPU inference libraries; no compiler is kept in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 regulatrace \
    && useradd --uid 1000 --gid regulatrace --create-home regulatrace \
    && install -d -o regulatrace -g regulatrace \
        /home/regulatrace/.cache \
        /home/regulatrace/.cache/huggingface /home/regulatrace/.cache/tiktoken

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/regulatrace \
    XDG_CACHE_HOME=/home/regulatrace/.cache \
    HF_HOME=/home/regulatrace/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/home/regulatrace/.cache/tiktoken \
    PYTHONPATH=/app/src \
    REGULATRACE_ENVIRONMENT=production \
    REGULATRACE_FRONTEND_DIST=/app/frontend/dist \
    REGULATRACE_CORS_ALLOWED_ORIGINS=[] \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

COPY --from=builder /opt/venv /opt/venv
# Source imports preserve Path(__file__).parents[2] / "prompts" in the app.
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY data/manifest.csv ./data/manifest.csv
COPY --from=frontend-builder /web/dist/ ./frontend/dist/

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + (os.environ.get('PORT') or '8000') + '/health', timeout=3).close()"]

# Run migrations separately as a release task: python -m alembic upgrade head.
# Configuration checks fail before a misleading healthy-but-unusable deployment.
CMD ["sh", "-c", "python -m regulatrace.deployment && exec python -m uvicorn regulatrace.api.app:create_app --factory --host 0.0.0.0 --port \"${PORT:-8000}\" --limit-concurrency 8"]
