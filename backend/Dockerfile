# Backend Dockerfile for Research Report Processor
# Base image: Python 3.11 slim for minimal size
#
# Build context: This Dockerfile should be built from the project root directory
# Example: docker build -f backend/Dockerfile -t research-report-processor-backend .

FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
# pyproject.toml is in the project root
COPY pyproject.toml README.md ./

# Copy backend source for package installation
COPY backend/ ./backend/

# Install dependencies (production only, no dev dependencies)
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy backend source code (excluding tests via .dockerignore)
COPY backend/ ./backend/

# Create storage directory and set permissions
RUN mkdir -p /app/storage && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port 8765 (configurable via SERVER_PORT env var)
EXPOSE 8765

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

# Run with uvicorn
# Port can be overridden via environment variable in docker-compose
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8765"]
