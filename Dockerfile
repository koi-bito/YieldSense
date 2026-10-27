# Multi-stage Dockerfile for YieldSense REST API Service
# Builder Stage
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Final Runtime Stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root system user for security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code, models, and data
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/
COPY pyproject.toml /app/

# Create reports directory with appropriate permissions
RUN mkdir -p /app/reports && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DB_PATH=/app/data/yieldsense.duckdb \
    MODEL_DIR=/app/models/

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
