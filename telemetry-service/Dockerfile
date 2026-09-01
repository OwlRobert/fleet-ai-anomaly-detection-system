# Telemetry Service — ingestion, normalization, persistence and history queries.
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE keeps .pyc files out of the image layer; PYTHONUNBUFFERED
# makes uvicorn's logs appear immediately in `docker compose logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, so editing application code does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service runs. Tests and local caches stay out, and .dockerignore
# keeps them out of the build context in the first place.
COPY app ./app

# Run unprivileged. Nothing here writes to disk at runtime.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Readiness, not liveness: this instance is only useful once its mandatory
# dependency — the telemetry store — answers. The Inference Service is
# deliberately not part of it, because ingestion is fail-open on inference.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
