FROM python:3.11-slim

# System deps for rasterio (GDAL) and psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libgdal-dev \
    gdal-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# AWS Batch: detect AWS context via env var to choose private DB host
# (set AWS_BATCH_JOB_ID automatically by Batch; POSTGIS_HOST_PRIVATE must be
#  configured in the Batch Job Definition environment section)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default entrypoint — AWS Batch overrides CMD via Job Definition
CMD ["python", "scripts/run_datacube_batch.py"]

# ── Health-check for local Docker testing ─────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from app.core.config import get_settings; s = get_settings(); print('OK')" \
    || exit 1
