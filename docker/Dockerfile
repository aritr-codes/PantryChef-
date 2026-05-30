# Serving image (Phase 6 placeholder). Multi-stage, uv-based.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen || uv sync --no-dev

COPY pantrychef ./pantrychef

# Phase 6: replace with the FastAPI entrypoint.
CMD ["uv", "run", "python", "-m", "pantrychef.cli"]
