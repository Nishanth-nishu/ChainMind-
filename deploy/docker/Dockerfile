# =============================================================================
# ChainMind — Multi-stage Docker build
# =============================================================================
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir --prefix=/install .

# =============================================================================
# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r chainmind && useradd -r -g chainmind -d /app -s /sbin/nologin chainmind

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY chainmind/ ./chainmind/
COPY data/ ./data/

# Create data directories
RUN mkdir -p /app/data/chromadb /app/data && chown -R chainmind:chainmind /app

# Switch to non-root user
USER chainmind

# Environment
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV LOG_FORMAT=json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()" || exit 1

# Expose ports
EXPOSE 8000 50051

# Default: run FastAPI
CMD ["python", "-m", "chainmind.api.app"]
