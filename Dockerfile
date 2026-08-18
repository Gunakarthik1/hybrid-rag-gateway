# ── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for healthcheck only
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/* && \
    rm -rf /wheels

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
