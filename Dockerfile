# Multi-stage, slim, non-root Dockerfile for VisionGuard (23L-0994_A2_Ops)
# Task 6.2 & Task 6.4: Optimized containerization with layer caching and CPU TensorFlow

# Base layer: Official slim Python Debian image
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

# Install minimal OS dependencies for curl (used in HEALTHCHECK) and clean apt cache immediately
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group with UID 1000 for least-privilege security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Step 1: Install Python dependencies BEFORE copying source code.
# This maximizes Docker layer caching: rebuilding after source edits will not reinstall packages.
COPY requirements.txt .

# Install CPU-only TensorFlow and runtime dependencies without pip wheel cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy application source code, assets, and templates
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
RUN mkdir -p /app/models

# Ensure application files and working directory are owned by non-root appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 5000

# Task 6.2 Healthcheck: Queries /health endpoint periodically
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server command
CMD ["python3", "src/app.py"]
